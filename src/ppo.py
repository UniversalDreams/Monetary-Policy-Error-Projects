import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config

def flatten_obs(obs_dict, device):
    """
    Crush the dictionary from DummyVecEnv into a single flat tensor
    """
    llm = torch.Tensor(obs_dict["llm_belief"]).to(device)
    macro = torch.Tensor(obs_dict["macro"]).to(device)
    return torch.cat([llm, macro], dim=-1)


class RolloutBuffer:
    """
    Stores data from the env interactions to be used in the PPO update
    """
    def __init__(self, num_steps, num_envs, obs_shape, action_shape, device):
        self.obs = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + action_shape).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)

        self.advantages = torch.zeros((num_steps, num_envs)).to(device)
        self.returns = torch.zeros((num_steps, num_envs)).to(device)

        self.step = 0
        self.num_steps = num_steps
        self.device = device

    def add(self, obs, action, logprob, reward, value, done):
        """
        Add a transition to the buffer at current step
        """
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.dones[self.step] = done

        self.step += 1

    def compute_returns_and_advantages(self, last_value, last_done, gamma=0.99, gae_lambda=0.95):
        """
        Computes GAE
        """
        last_gae_lam = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            # td error
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]

            # advantage
            self.advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        self.returns = self.advantages + self.values

    def get_generator(self, batch_size):
        """
        Get mini-batches of data for the PPO update epochs
        """
        num_steps_taken = self.num_steps
        num_environments = self.rewards.shape[1]
        total_experiences = num_steps_taken * num_environments
        shuffled_indices = np.random.permutation(total_experiences)

        # flatten the historical data
        # use self.obs.shape[2:] to keep the shape of the observation itself
        obs_feature_shape = self.obs.shape[2:]
        flat_obs = self.obs.view(-1, *obs_feature_shape)

        action_feature_shape = self.actions.shape[2:]
        flat_actions = self.actions.view(-1, *action_feature_shape)

        flat_logprobs = self.logprobs.view(-1)
        flat_advantages = self.advantages.view(-1)
        flat_returns = self.returns.view(-1)

        # iterate through the shuffled indices
        for start_idx in range(0, total_experiences, batch_size):
            end_idx = start_idx + batch_size
            current_batch_indices = shuffled_indices[start_idx:end_idx]

            yield (
                flat_obs[current_batch_indices],
                flat_actions[current_batch_indices],
                flat_logprobs[current_batch_indices],
                flat_advantages[current_batch_indices],
                flat_returns[current_batch_indices]
            )

class PPOAgent:
    """
    Main PPO Algo loop
    """
    def __init__(self, envs, actor_critic, device):
        self.envs = envs
        self.actor_critic = actor_critic
        self.device = device

        # hyperparameters from config.py
        self.lr = config.LR
        self.num_steps = config.N_STEPS
        self.batch_size = config.BATCH_SIZE
        self.n_epochs = config.N_EPOCHS
        self.gamma = config.GAMMA
        self.clip_range = 0.2
        self.ent_coef_start = config.BASELINE_ENT_COEF
        self.ent_coef = self.ent_coef_start

        # init optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=1e-5)

        # buffer setup - manually calculate total dims for the dict space
        macro_dim = envs.observation_space["macro"].shape[0]
        llm_dim = envs.observation_space["llm_belief"].shape[0]
        obs_shape = (macro_dim + llm_dim,)
        action_shape = envs.action_space.shape

        self.buffer = RolloutBuffer(self.num_steps, config.N_ENVS, obs_shape, action_shape, device)

    def learn(self, total_timesteps, cond_dir=None):
        """
        Main training loop
        """
        # setup csv logging
        csv_file = None
        csv_writer = None
        if cond_dir:
            csv_path = os.path.join(cond_dir, "training.csv")
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "episode", "total_steps", "ep_reward",
                "avg_P_supply", "avg_hawkishness", "avg_uncertainty",
                "ent_coef", "elapsed_s"
            ])

        # track total steps
        global_step = 0

        # get init obs from vectorized envs and flatten it
        raw_obs = self.envs.reset()
        next_obs = flatten_obs(raw_obs, self.device)

        # init done flags for envs
        next_done = torch.zeros(config.N_ENVS).to(self.device)

        # calc total ppo updates to run
        num_updates = total_timesteps // (self.num_steps * config.N_ENVS)

        # tracking for logs
        episodes_completed = 0
        start_time = time.time()
        best_ep_reward = -np.inf
        best_ema = -np.inf
        current_ema = None
        alpha = 2.0 / (100 + 1)

        # accumulators for the 4 parallel environments
        current_ep_rewards = np.zeros(config.N_ENVS)
        current_ep_steps = np.zeros(config.N_ENVS)
        current_ep_P_supply = np.zeros(config.N_ENVS)
        current_ep_hawkishness = np.zeros(config.N_ENVS)
        current_ep_uncertainty = np.zeros(config.N_ENVS)

        for update in range(1, num_updates + 1):
            # rollout
            for step in range(0, self.num_steps):
                global_step += config.N_ENVS

                with torch.no_grad():
                    action, logprob, value = self.actor_critic.get_action_and_value(next_obs)

                # step in vectorized environment
                raw_new_obs, rewards, dones, infos = self.envs.step(action.cpu().numpy())

                # extract llm beliefs from the observation dict
                # Shape: (N_ENVS, 5) -> [P_normal, P_supply, sentiment, hawkishness, uncertainty]
                llm_beliefs = raw_new_obs.get("llm_belief", np.zeros((config.N_ENVS, 5)))

                # accumulate metrics
                current_ep_rewards += rewards
                current_ep_steps += 1
                current_ep_P_supply += llm_beliefs[:, 1]
                current_ep_hawkishness += llm_beliefs[:, 3]
                current_ep_uncertainty += llm_beliefs[:, 4]

                for i, done in enumerate(dones):
                    if done:
                        episodes_completed += 1
                        ep_reward = float(current_ep_rewards[i])

                        # calculate episode averages
                        steps_taken = max(1, int(current_ep_steps[i]))
                        avg_P_supply = float(current_ep_P_supply[i]) / steps_taken
                        avg_hawkishness = float(current_ep_hawkishness[i]) / steps_taken
                        avg_uncertainty = float(current_ep_uncertainty[i]) / steps_taken
                        elapsed = time.time() - start_time

                        # log to csv and terminal
                        if csv_writer:
                            csv_writer.writerow([
                                episodes_completed, global_step, round(ep_reward, 4),
                                round(avg_P_supply, 4), round(avg_hawkishness, 4),
                                round(avg_uncertainty, 4), round(self.ent_coef, 6),
                                round(elapsed, 1)
                            ])
                            csv_file.flush()

                        if cond_dir and ep_reward > best_ep_reward:
                            best_ep_reward = ep_reward
                            best_path = os.path.join(cond_dir, "best_model_weights.pth")
                            torch.save(self.actor_critic.state_dict(), best_path)

                        if current_ema is None:
                            current_ema = ep_reward
                        else:
                            current_ema = alpha * ep_reward + (1 - alpha) * current_ema

                        if cond_dir and current_ema > best_ema:
                            best_ema = current_ema
                            ema_path = os.path.join(cond_dir, "best_model_ema_weights.pth")
                            torch.save(self.actor_critic.state_dict(), ema_path)

                        if episodes_completed % 50 == 0:
                            print(f"Update {update}/{num_updates} | Ep {episodes_completed} | "
                                  f"Reward: {ep_reward:+.2f} | P_sup: {avg_P_supply:.2f} | "
                                  f"Unc: {avg_uncertainty:.2f} | Time: {elapsed:.0f}s")

                        # reset trackers for this specific environment
                        current_ep_rewards[i] = 0.0
                        current_ep_steps[i] = 0.0
                        current_ep_P_supply[i] = 0.0
                        current_ep_hawkishness[i] = 0.0
                        current_ep_uncertainty[i] = 0.0

                # convert results to tensors
                rewards_tensor = torch.Tensor(rewards).to(self.device)
                dones_tensor = torch.Tensor(dones).to(self.device)

                # store data in buffer
                self.buffer.add(next_obs, action, logprob, rewards_tensor, value.flatten(), next_done)

                # update state for next step
                next_obs = flatten_obs(raw_new_obs, self.device)
                next_done = dones_tensor

            # advantage estimation
            with torch.no_grad():
                # bootstrap the next value of the final obs for the GAE calculation
                _, _, next_value = self.actor_critic.get_action_and_value(next_obs)

            self.buffer.compute_returns_and_advantages(next_value.flatten(), next_done, gamma=self.gamma)

            # calculate progress and decay entropy coefficient linearly
            progress = (update - 1) / num_updates
            self.ent_coef = self.ent_coef_start * (1.0 - progress)

            # update
            self._update_policy()

            # reset buffer
            self.buffer.step = 0

        if csv_file:
            csv_file.close()

    def _update_policy(self):
        """
        Runs the PPO epochs and mini-batch updates to optimize the policy
        """
        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_generator(self.batch_size):
                b_obs, b_actions, b_old_logprobs, b_advantages, b_returns = batch

                # normalize advantages per mini-batch
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                # evaluate the historical actions on the historical states
                new_logprobs, entropy, new_values = self.actor_critic.evaluate_actions(b_obs, b_actions)

                # calculate the ratio
                logratio = new_logprobs - b_old_logprobs
                ratio = logratio.exp()

                # calculate policy loss
                surrogate_1 = ratio * b_advantages
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * b_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # value loss
                value_loss = nn.MSELoss()(new_values.squeeze(-1), b_returns)

                # entropy loss for exploration
                entropy_loss = -entropy.mean()

                # total loss
                loss = policy_loss + 0.5 * value_loss + self.ent_coef * entropy_loss

                # back and optimize
                self.optimizer.zero_grad()
                loss.backward()

                # prevent exploding gradients
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)

                self.optimizer.step()
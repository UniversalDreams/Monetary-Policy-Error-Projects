import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config

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

            batch_obs = flat_obs[current_batch_indices]
            batch_actions = flat_actions[current_batch_indices]
            batch_logprobs = flat_logprobs[current_batch_indices]
            batch_advantages = flat_advantages[current_batch_indices]
            batch_returns = flat_returns[current_batch_indices]

            yield (
                batch_obs,
                batch_actions,
                batch_logprobs,
                batch_advantages,
                batch_returns
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
        self.clip_range = config.BASELINE_CLIP_RANGE
        self.ent_coef = config.BASELINE_ENT_COEF

        # init optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=1e-5)

        # buffer
        obs_shape = envs.single_observation_space.shape
        action_shape = envs.single_action_space.shape
        self.buffer = RolloutBuffer(self.num_steps, config.N_ENVS, obs_shape, action_shape, device)

    def learn(self, total_timesteps):
        """
        Main training loop
        """
        # track total steps
        global_step = 0

        # get init obs from vectorized envs
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)

        # init done flags for envs
        next_done = torch.zeros(config.N_ENVS).to(self.device)

        # calc total ppo updates to run
        num_updates = total_timesteps // (self.num_steps * config.N_ENVS)

        for update in range(1, num_updates + 1):
            # rollout
            for step in range(0, self.num_steps):
                global_step += config.N_ENVS

                with torch.no_grad():
                    action, logprob, _, value = self.actor_critic.get_action_and_value(next_obs)

                # step in vectorized environment
                obs, rewards, dones, infos = self.envs.step(action.cpu().numpy())

                # convert results to tensors
                rewards_tensor = torch.Tensor(rewards).to(self.device)
                dones_tensor = torch.Tensor(dones).to(self.device)

                # store data in buffer
                self.buffer.add(next_obs, action, logprob, rewards_tensor, value.flatten(), next_done)

                # update state for next step
                next_obs = torch.Tensor(obs).to(self.device)
                next_done = dones_tensor

            # advantage estimation
            with torch.no_grad():
                # bootstrap the next value of the final obs for the GAE calculation
                _, _, _, next_value = self.actor_critic.get_action_and_value(next_obs)

            self.buffer.compute_returns_and_advantages(next_value.flatten(), next_done, gamma=self.gamma)

            # update
            self._update_policy()

            # reset buffer
            self.buffer.step = 0

    def _update_policy(self):
        """
        Runs the PPO epochs and mini-batch updates to optimize the policy
        """
        # normalize advantages across the entire rollout batch
        flat_advs = self.buffer.advantages.view(-1)
        normalized_advs = (flat_advs - flat_advs.mean()) / (flat_advs.std() + 1e-8)
        self.buffer.advantages = normalized_advs.view(self.buffer.advantages.shape)

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_generator(self.batch_size):
                b_obs, b_actions, b_old_logprobs, b_advantages, b_returns = batch

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
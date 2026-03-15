import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import config

from drqn_network import DuelingDRQN
from drqn_buffer import SequentialReplayBuffer


def flatten_obs(obs_dict, device):
    """Crush the dictionary from DummyVecEnv into a single flat tensor"""
    llm = torch.Tensor(obs_dict["llm_belief"]).to(device)
    macro = torch.Tensor(obs_dict["macro"]).to(device)
    return torch.cat([llm, macro], dim=-1)


class DRQNAgent:
    def __init__(self, envs, device):
        self.envs = envs
        self.device = device

        # dimensions
        macro_dim = envs.observation_space["macro"].shape[0]
        llm_dim = envs.observation_space["llm_belief"].shape[0]
        self.obs_dim = macro_dim + llm_dim
        self.act_dim = envs.action_space.n
        self.n_envs = config.N_ENVS

        # hyperparameters
        self.lr = 1e-4
        self.gamma = config.GAMMA
        self.tau = 0.005  # soft update rate for target network
        self.batch_size = 32
        self.seq_len = 8  # length of sequences to sample from buffer
        self.update_freq = 4  # steps between network updates

        # epsilon-greedy exploration schedule
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 500_000
        self.epsilon = self.epsilon_start

        # networks
        self.policy_net = DuelingDRQN(self.obs_dim, self.act_dim).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # Huber Loss

        # replay buffer
        self.buffer = SequentialReplayBuffer(capacity_episodes=2000, seq_len=self.seq_len, obs_dim=self.obs_dim)

    def select_action(self, obs, hidden_state):
        """Epsilon-greedy action selection for N parallel environments."""
        if np.random.rand() < self.epsilon:
            # random actions for exploration
            actions = np.random.randint(0, self.act_dim, size=self.n_envs)

            with torch.no_grad():
                _, next_hidden = self.policy_net(obs.unsqueeze(1), hidden_state)
            return actions, next_hidden

        else:
            # exploitation
            with torch.no_grad():
                # obs shape: (N_ENVS, ObsDim) -> add seq_len dim: (N_ENVS, 1, ObsDim)
                q_values, next_hidden = self.policy_net(obs.unsqueeze(1), hidden_state)

                # q_values shape: (N_ENVS, 1, ActDim) -> get argmax
                actions = q_values.squeeze(1).argmax(dim=-1).cpu().numpy()
            return actions, next_hidden

    def update(self):
        """Samples a batch of sequences and performs Double Q-Learning update."""
        if len(self.buffer) < self.batch_size:
            return  # wait until we have enough episodes to train

        b_obs, b_acts, b_rews, b_next_obs, b_dones, b_mask = self.buffer.sample(self.batch_size, self.device)

        # initialize zero hidden states for the batch
        h_0, c_0 = self.policy_net.get_initial_hidden_state(self.batch_size, self.device)

        # get current Q-values for the actions we actually took
        current_q_all, _ = self.policy_net(b_obs, (h_0, c_0))
        # gather the Q-values for the specific actions
        current_q = current_q_all.gather(-1, b_acts).squeeze(-1)  # Shape: (Batch, Seq)

        with torch.no_grad():
            # use policy net to select the best action for the next state
            next_q_policy, _ = self.policy_net(b_next_obs, (h_0, c_0))
            best_next_actions = next_q_policy.argmax(dim=-1, keepdim=True)

            # use target net to evaluate that selected action
            next_q_target_all, _ = self.target_net(b_next_obs, (h_0, c_0))
            next_q = next_q_target_all.gather(-1, best_next_actions).squeeze(-1)

            # calculate target
            target_q = b_rews + (self.gamma * (1 - b_dones) * next_q)

        # calculate loss, applying the mask so we ignore padded elements
        loss = self.loss_fn(current_q, target_q)
        masked_loss = (loss * b_mask).sum() / b_mask.sum()

        # optimize
        self.optimizer.zero_grad()
        masked_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        self.optimizer.step()

        # soft-update Target Network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        return {
            "policy_loss": masked_loss.item(),
            "value_loss": 0.0,
            "entropy_loss": 0.0
        }

    def learn(self, total_timesteps, on_episode=None, on_checkpoint=None, on_update=None):
        """Main training loop."""
        num_iterations = total_timesteps // self.n_envs

        self.epsilon_decay_steps = int(num_iterations * 0.25)

        raw_obs = self.envs.reset()
        obs = flatten_obs(raw_obs, self.device)

        # initialize hidden state for N parallel environments
        hidden_state = self.policy_net.get_initial_hidden_state(self.n_envs, self.device)

        ep_obs, ep_acts, ep_rews, ep_dones = [[] for _ in range(self.n_envs)], [[] for _ in range(self.n_envs)], [[] for _ in range(self.n_envs)], [[] for _ in range(self.n_envs)]

        # track raw rewards
        ep_raw_reward_sums = np.zeros(self.n_envs)

        for step in range(num_iterations): # <-- Use num_iterations here!
            global_step = step * self.n_envs # <-- Calculate true global steps for logging

            # act
            actions, next_hidden_state = self.select_action(obs, hidden_state)

            # step environment
            raw_new_obs, rewards, dones, infos = self.envs.step(actions)
            next_obs = flatten_obs(raw_new_obs, self.device)

            # extract original rewards
            try:
                original_rewards = self.envs.get_original_reward()
            except AttributeError:
                original_rewards = rewards

            # store transitions for each env
            for i in range(self.n_envs):
                ep_obs[i].append(obs[i].cpu().numpy())
                ep_acts[i].append(actions[i])
                ep_rews[i].append(rewards[i])
                ep_dones[i].append(float(dones[i]))
                ep_raw_reward_sums[i] += original_rewards[i]

                # if an environment finished an episode
                if dones[i]:
                    # push the completed trajectory to the sequential buffer
                    self.buffer.push_episode(ep_obs[i], ep_acts[i], ep_rews[i], ep_dones[i])

                    if on_episode:
                        llm_belief = raw_new_obs["llm_belief"][i:i+1] # Shape (1, 5)
                        ep_norm_rew = sum(ep_rews[i])
                        # Pass true global_step to align your CSVs!
                        on_episode(ep_raw_reward_sums[i], ep_norm_rew, global_step, self.epsilon, llm_belief)

                    # reset trackers for this env
                    ep_obs[i], ep_acts[i], ep_rews[i], ep_dones[i] = [], [], [], []
                    ep_raw_reward_sums[i] = 0.0

                    h, c = next_hidden_state
                    h[:, i, :] = 0.0
                    c[:, i, :] = 0.0
                    next_hidden_state = (h, c)

            # update state variables
            obs = next_obs
            hidden_state = next_hidden_state

            # decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon_start - (step / self.epsilon_decay_steps))

            # train the network
            if step % self.update_freq == 0:
                losses = self.update()
                if on_update and losses is not None:
                    on_update(losses, global_step)

            if on_checkpoint and step % (config.CHECKPOINT_FREQ // self.n_envs) == 0:
                on_checkpoint(self.policy_net, global_step)
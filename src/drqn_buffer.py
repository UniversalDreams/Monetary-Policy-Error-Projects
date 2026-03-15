import numpy as np
import torch


class SequentialReplayBuffer:
    """
    Replay buffer that stores full episodes and samples sequences of length `seq_len`
    for Recurrent Q-Learning.
    """

    def __init__(self, capacity_episodes=1000, seq_len=8, obs_dim=8):
        self.capacity = capacity_episodes
        self.seq_len = seq_len
        self.obs_dim = obs_dim

        self.buffer = []
        self.position = 0

    def push_episode(self, episode_obs, episode_acts, episode_rews, episode_dones):
        """
        Pushes a full episode to the buffer.
        """
        # convert to numpy arrays
        ep_obs = np.array(episode_obs, dtype=np.float32)
        ep_acts = np.array(episode_acts, dtype=np.int64)
        ep_rews = np.array(episode_rews, dtype=np.float32)
        ep_dones = np.array(episode_dones, dtype=np.float32)

        episode_data = {
            'obs': ep_obs,
            'acts': ep_acts,
            'rews': ep_rews,
            'dones': ep_dones
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(episode_data)
        else:
            self.buffer[self.position] = episode_data

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device):
        """
        Samples a batch of sequences of length `seq_len`.
        """
        # randomly choose `batch_size` episodes
        indices = np.random.randint(0, len(self.buffer), batch_size)

        # pre-allocate batch tensors
        batch_obs = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        batch_acts = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        batch_rews = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        batch_next_obs = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        batch_dones = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        # mask to ignore padded steps if an episode was shorter than seq_len
        batch_mask = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        for b, idx in enumerate(indices):
            ep = self.buffer[int(idx)]
            ep_len = len(ep['obs'])

            # pick a random starting transition
            max_start = max(0, ep_len - self.seq_len - 1)
            start_t = np.random.randint(0, max_start + 1)

            end_t = min(start_t + self.seq_len, ep_len - 1)
            actual_seq_len = end_t - start_t

            # fill the batch arrays
            batch_obs[b, :actual_seq_len] = ep['obs'][start_t:end_t]
            batch_acts[b, :actual_seq_len] = ep['acts'][start_t:end_t]
            batch_rews[b, :actual_seq_len] = ep['rews'][start_t:end_t]
            batch_next_obs[b, :actual_seq_len] = ep['obs'][start_t + 1:end_t + 1]
            batch_dones[b, :actual_seq_len] = ep['dones'][start_t:end_t]
            batch_mask[b, :actual_seq_len] = 1.0

        return (
            torch.tensor(batch_obs).to(device),
            torch.tensor(batch_acts).unsqueeze(-1).to(device),
            torch.tensor(batch_rews).to(device),
            torch.tensor(batch_next_obs).to(device),
            torch.tensor(batch_dones).to(device),
            torch.tensor(batch_mask).to(device)
        )

    def __len__(self):
        return len(self.buffer)
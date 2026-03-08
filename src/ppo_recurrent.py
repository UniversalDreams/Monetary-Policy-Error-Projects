"""
src/ppo_recurrent.py — Recurrent PPO with sequence-aware rollout buffer.

Provides RecurrentRolloutBuffer and RecurrentPPOAgent with the same external
interface as RolloutBuffer / PPOAgent in ppo.py, so train_custom.py can call
agent.learn(total_timesteps) identically for both mlp and lstm policies.
"""

import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

import config
from ppo import flatten_obs


class RecurrentRolloutBuffer:
    """
    Rollout buffer that stores LSTM hidden states and episode-start flags
    alongside the standard PPO data, enabling sequence-aware mini-batching.
    """

    def __init__(self, num_steps, num_envs, obs_shape, action_shape,
                 n_lstm_layers, lstm_hidden_size, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.n_lstm_layers = n_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.device = device

        # Standard PPO storage
        self.obs        = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
        self.actions    = torch.zeros((num_steps, num_envs) + action_shape).to(device)
        self.logprobs   = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards    = torch.zeros((num_steps, num_envs)).to(device)
        self.values     = torch.zeros((num_steps, num_envs)).to(device)
        self.dones      = torch.zeros((num_steps, num_envs)).to(device)

        self.advantages = torch.zeros((num_steps, num_envs)).to(device)
        self.returns    = torch.zeros((num_steps, num_envs)).to(device)

        # Recurrent-specific storage
        # episode_starts[t, env] = 1.0 if step t begins a new episode for env
        self.episode_starts = torch.zeros((num_steps, num_envs)).to(device)

        # Hidden states BEFORE the LSTM processed step t  (shape mirrors LSTM state)
        # stored as (num_steps, n_layers, num_envs, hidden_size)
        h_shape = (num_steps, n_lstm_layers, num_envs, lstm_hidden_size)
        self.hidden_states_pi = torch.zeros(h_shape).to(device)
        self.cell_states_pi   = torch.zeros(h_shape).to(device)
        self.hidden_states_vf = torch.zeros(h_shape).to(device)
        self.cell_states_vf   = torch.zeros(h_shape).to(device)

        self.step = 0

    def add(self, obs, action, logprob, reward, value, done, hidden, episode_starts):
        """
        Store one timestep of data for all envs.

        Args:
            obs:             (n_envs, obs_dim)
            action:          (n_envs,)
            logprob:         (n_envs,)
            reward:          (n_envs,)
            value:           (n_envs,)
            done:            (n_envs,)
            hidden:          (h_pi, c_pi, h_vf, c_vf) or None
            episode_starts:  (n_envs,) — 1.0 if this step begins a new episode
        """
        self.obs[self.step]            = obs
        self.actions[self.step]        = action
        self.logprobs[self.step]       = logprob
        self.rewards[self.step]        = reward
        self.values[self.step]         = value
        self.dones[self.step]          = done
        self.episode_starts[self.step] = episode_starts

        if hidden is not None:
            h_pi, c_pi, h_vf, c_vf = hidden
            self.hidden_states_pi[self.step] = h_pi  # (n_layers, n_envs, hidden_size)
            self.cell_states_pi[self.step]   = c_pi
            self.hidden_states_vf[self.step] = h_vf
            self.cell_states_vf[self.step]   = c_vf
        # If hidden is None, the pre-zeroed buffer values (zeros) are correct.

        self.step += 1

    def compute_returns_and_advantages(self, last_value, last_done, gamma=0.99, gae_lambda=0.95):
        """Computes GAE advantages and returns. Identical to RolloutBuffer."""
        last_gae_lam = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            self.advantages[step] = last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )

        self.returns = self.advantages + self.values

    def get_generator(self, batch_size):
        """
        Sequence-aware generator. Splits each env's trajectory at episode boundaries
        into segments, shuffles, groups into batches of ~batch_size total timesteps,
        pads to uniform length, and yields 8-tuples.

        Yields:
            b_obs:         (n_seq * max_len, obs_dim)
            b_actions:     (n_seq * max_len,)
            b_old_logprobs:(n_seq * max_len,)
            b_advantages:  (n_seq * max_len,)
            b_returns:     (n_seq * max_len,)
            b_lstm_states: (h_pi, c_pi, h_vf, c_vf) each (n_layers, n_seq, hidden_size)
            b_ep_starts:   (n_seq * max_len,)  — 1.0 at position 0 of each sequence
            b_mask:        (n_seq * max_len,) bool — True for real timesteps
        """
        # --- 1. Identify sequence segments ---
        # A new segment starts at step 0 or wherever episode_starts == 1
        segments = []  # (env_idx, t_start, t_end)
        for env_idx in range(self.num_envs):
            t_start = 0
            for t in range(1, self.num_steps):
                if self.episode_starts[t, env_idx] > 0.5:
                    segments.append((env_idx, t_start, t))
                    t_start = t
            segments.append((env_idx, t_start, self.num_steps))

        # --- 2. Shuffle with split trick ---
        n_segs = len(segments)
        split_idx = np.random.randint(n_segs)
        order = list(range(split_idx, n_segs)) + list(range(0, split_idx))

        # --- 3. Group segments into batches by total timestep count ---
        current_batch: list = []
        current_size = 0

        for idx in order:
            seg = segments[idx]
            seg_len = seg[2] - seg[1]
            if current_batch and current_size + seg_len > batch_size:
                yield self._make_batch(current_batch)
                current_batch = []
                current_size = 0
            current_batch.append(seg)
            current_size += seg_len

        if current_batch:
            yield self._make_batch(current_batch)

    def _make_batch(self, segs):
        """Build and return one padded batch from a list of (env_idx, t_start, t_end) segments."""
        seqs_obs, seqs_act, seqs_logp = [], [], []
        seqs_adv, seqs_ret, seqs_ep   = [], [], []
        init_h_pi, init_c_pi = [], []
        init_h_vf, init_c_vf = [], []
        seq_lens = []

        for env_idx, t_start, t_end in segs:
            seq_len = t_end - t_start
            seq_lens.append(seq_len)

            seqs_obs.append(self.obs[t_start:t_end, env_idx])          # (seq_len, obs_dim)
            seqs_act.append(self.actions[t_start:t_end, env_idx])       # (seq_len,) or (seq_len, act_dim)
            seqs_logp.append(self.logprobs[t_start:t_end, env_idx])     # (seq_len,)
            seqs_adv.append(self.advantages[t_start:t_end, env_idx])    # (seq_len,)
            seqs_ret.append(self.returns[t_start:t_end, env_idx])       # (seq_len,)

            # Episode starts: first step of sequence always resets LSTM state
            ep_s = torch.zeros(seq_len, device=self.device)
            ep_s[0] = 1.0
            seqs_ep.append(ep_s)

            # Initial hidden state at the START of this sequence
            # hidden_states shape: (num_steps, n_layers, num_envs, hidden_size)
            # → [t_start, :, env_idx, :] = (n_layers, hidden_size)
            init_h_pi.append(self.hidden_states_pi[t_start, :, env_idx, :])
            init_c_pi.append(self.cell_states_pi[t_start, :, env_idx, :])
            init_h_vf.append(self.hidden_states_vf[t_start, :, env_idx, :])
            init_c_vf.append(self.cell_states_vf[t_start, :, env_idx, :])

        n_seq = len(segs)

        # --- Pad sequences (batch_first=True → (n_seq, max_len, ...)) ---
        padded_obs  = pad_sequence(seqs_obs,  batch_first=True)  # (n_seq, max_len, obs_dim)
        padded_act  = pad_sequence(seqs_act,  batch_first=True)  # (n_seq, max_len[, act_dim])
        padded_logp = pad_sequence(seqs_logp, batch_first=True)  # (n_seq, max_len)
        padded_adv  = pad_sequence(seqs_adv,  batch_first=True)  # (n_seq, max_len)
        padded_ret  = pad_sequence(seqs_ret,  batch_first=True)  # (n_seq, max_len)
        padded_ep   = pad_sequence(seqs_ep,   batch_first=True)  # (n_seq, max_len)

        max_len = padded_obs.shape[1]

        # Boolean mask: True for real timesteps, False for padding
        mask = torch.zeros(n_seq, max_len, dtype=torch.bool, device=self.device)
        for i, sl in enumerate(seq_lens):
            mask[i, :sl] = True

        # Stack initial hidden states: list of (n_layers, hidden_size) → (n_layers, n_seq, hidden_size)
        def stack_hidden(lst):
            return torch.stack(lst, dim=0).permute(1, 0, 2)  # (n_layers, n_seq, hidden_size)

        b_lstm_states = (
            stack_hidden(init_h_pi),
            stack_hidden(init_c_pi),
            stack_hidden(init_h_vf),
            stack_hidden(init_c_vf),
        )

        # Flatten: (n_seq, max_len, ...) → (n_seq * max_len, ...)
        def flat(t):
            return t.reshape(n_seq * max_len, *t.shape[2:])

        return (
            flat(padded_obs),
            flat(padded_act),
            flat(padded_logp),
            flat(padded_adv),
            flat(padded_ret),
            b_lstm_states,
            flat(padded_ep),
            mask.reshape(n_seq * max_len),
        )


class RecurrentPPOAgent:
    """
    PPO agent that uses LSTMActorCritic and RecurrentRolloutBuffer.
    Exposes the same learn(total_timesteps) interface as PPOAgent.
    """

    def __init__(self, envs, actor_critic, device):
        self.envs         = envs
        self.actor_critic = actor_critic
        self.device       = device

        # Hyperparameters (LSTM variants from config)
        self.lr           = config.LR
        self.num_steps    = config.LSTM_N_STEPS
        self.batch_size   = config.LSTM_BATCH_SIZE
        self.n_epochs     = config.LSTM_N_EPOCHS
        self.gamma        = config.GAMMA
        self.clip_range   = config.BASELINE_CLIP_RANGE
        self.ent_coef_start = config.BASELINE_ENT_COEF
        self.ent_coef     = self.ent_coef_start

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=self.lr, eps=1e-5)

        # Buffer shape
        macro_dim = envs.observation_space["macro"].shape[0]
        llm_dim   = envs.observation_space["llm_belief"].shape[0]
        obs_shape    = (macro_dim + llm_dim,)
        action_shape = envs.action_space.shape

        self.buffer = RecurrentRolloutBuffer(
            self.num_steps,
            config.N_ENVS,
            obs_shape,
            action_shape,
            actor_critic.lstm_actor.num_layers,
            actor_critic.lstm_actor.hidden_size,
            device,
        )

    def learn(self, total_timesteps, csv_log_path=None, tag="lstm"):
        """
        Main training loop — same signature as PPOAgent.learn().

        Args:
            total_timesteps: total env steps to train for
            csv_log_path:    optional path to write per-episode CSV log
            tag:             label used in console episode prints
        """
        global_step = 0

        raw_obs = self.envs.reset()
        next_obs = flatten_obs(raw_obs, self.device)
        next_done = torch.zeros(config.N_ENVS, device=self.device)

        # LSTM state carried across rollout windows
        self._last_hidden         = None
        self._last_episode_starts = torch.zeros(config.N_ENVS, device=self.device)

        num_updates = total_timesteps // (self.num_steps * config.N_ENVS)

        # Episode tracking
        ep_rewards = torch.zeros(config.N_ENVS, device=self.device)
        ep_count   = 0
        start_time = time.time()

        csv_file   = None
        csv_writer = None
        if csv_log_path:
            os.makedirs(os.path.dirname(csv_log_path) or ".", exist_ok=True)
            csv_file = open(csv_log_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["episode", "total_steps", "ep_reward", "elapsed_s"])

        try:
            for update in range(1, num_updates + 1):
                # --- Rollout collection ---
                for step in range(self.num_steps):
                    global_step += config.N_ENVS

                    with torch.no_grad():
                        action, logprob, value, new_hidden = self.actor_critic.get_action_and_value(
                            next_obs, self._last_hidden, self._last_episode_starts
                        )

                    raw_new_obs, rewards, dones, infos = self.envs.step(action.cpu().numpy())

                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                    dones_tensor   = torch.tensor(dones,   dtype=torch.float32, device=self.device)

                    # Store hidden state BEFORE this step and episode_starts for THIS step
                    self.buffer.add(
                        next_obs,
                        action,
                        logprob,
                        rewards_tensor,
                        value.flatten(),
                        dones_tensor,
                        self._last_hidden,           # hidden state at start of this step
                        self._last_episode_starts,   # 1.0 if this step opens a new episode
                    )

                    # Track episode rewards for logging
                    ep_rewards += rewards_tensor
                    for i, done in enumerate(dones):
                        if done:
                            ep_count += 1
                            elapsed = time.time() - start_time
                            print(
                                f"[{tag}] ep {ep_count}  reward={ep_rewards[i]:.2f}"
                                f"  steps={global_step}  elapsed={elapsed:.0f}s",
                                flush=True,
                            )
                            if csv_writer:
                                csv_writer.writerow([
                                    ep_count, global_step,
                                    round(float(ep_rewards[i]), 4),
                                    round(elapsed, 1),
                                ])
                                csv_file.flush()
                            ep_rewards[i] = 0.0

                    # Advance state
                    self._last_hidden         = new_hidden
                    self._last_episode_starts = dones_tensor
                    next_obs  = flatten_obs(raw_new_obs, self.device)
                    next_done = dones_tensor

                # --- Bootstrap value for GAE ---
                with torch.no_grad():
                    _, _, next_value, _ = self.actor_critic.get_action_and_value(
                        next_obs, self._last_hidden, self._last_episode_starts
                    )

                self.buffer.compute_returns_and_advantages(
                    next_value.flatten(), next_done, gamma=self.gamma
                )

                # Decay entropy coefficient
                progress = (update - 1) / num_updates
                self.ent_coef = self.ent_coef_start * (1.0 - progress)

                self._update_policy()

                # Reset buffer step counter (carry hidden state across windows)
                self.buffer.step = 0

        finally:
            if csv_file:
                csv_file.close()

    def _update_policy(self):
        """Runs PPO epochs with sequence-aware mini-batches."""
        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_generator(self.batch_size):
                (b_obs, b_actions, b_old_logprobs, b_advantages, b_returns,
                 b_lstm_states, b_ep_starts, b_mask) = batch

                # Normalize advantages over real (non-padded) timesteps only
                adv_real = b_advantages[b_mask]
                b_advantages = (b_advantages - adv_real.mean()) / (adv_real.std() + 1e-8)

                new_logprobs, entropy, new_values = self.actor_critic.evaluate_actions(
                    b_obs, b_actions, b_lstm_states, b_ep_starts
                )

                # PPO clipped policy loss (masked)
                logratio  = new_logprobs - b_old_logprobs
                ratio     = logratio.exp()
                surr1     = ratio * b_advantages
                surr2     = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * b_advantages
                policy_loss = -torch.min(surr1, surr2)[b_mask].mean()

                # Value loss (masked)
                value_loss = ((new_values.squeeze(-1) - b_returns) ** 2)[b_mask].mean()

                # Entropy loss (masked)
                entropy_loss = -entropy[b_mask].mean()

                loss = policy_loss + 0.5 * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()

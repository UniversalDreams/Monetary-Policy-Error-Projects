"""
src/ppo_recurrent.py — Recurrent PPO with sequence-aware rollout buffer.

Provides RecurrentRolloutBuffer and RecurrentPPOAgent with the same external
interface as RolloutBuffer / PPOAgent in ppo.py, so train_custom.py can call
agent.learn(total_timesteps) identically for both mlp and lstm policies.
"""

import os

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
        # Hidden states BEFORE the LSTM processed step t  (shape mirrors LSTM state)
        # stored as (num_steps, n_layers, num_envs, hidden_size)
        h_shape = (num_steps, n_lstm_layers, num_envs, lstm_hidden_size)
        self.hidden_states_pi = torch.zeros(h_shape).to(device)
        self.cell_states_pi   = torch.zeros(h_shape).to(device)
        self.hidden_states_vf = torch.zeros(h_shape).to(device)
        self.cell_states_vf   = torch.zeros(h_shape).to(device)

        self.step = 0

    def add(self, obs, action, logprob, reward, value, done, hidden):
        """
        Store one timestep of data for all envs.

        Args:
            obs:    (n_envs, obs_dim)
            action: (n_envs,)
            logprob:(n_envs,)
            reward: (n_envs,)
            value:  (n_envs,)
            done:   (n_envs,) — 1.0 if the PREVIOUS step ended an episode (episode boundary flag)
            hidden: (h_pi, c_pi, h_vf, c_vf) or None
        """
        self.obs[self.step]      = obs
        self.actions[self.step]  = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step]  = reward
        self.values[self.step]   = value
        self.dones[self.step]    = done

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
            b_old_values:  (n_seq * max_len,)
        """
        # --- 1. Identify sequence segments ---
        # A new segment starts at step 0 or wherever episode_starts == 1
        segments = []  # (env_idx, t_start, t_end)
        for env_idx in range(self.num_envs):
            t_start = 0
            for t in range(1, self.num_steps):
                if self.dones[t, env_idx] > 0.5:
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
        seqs_val = []
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
            seqs_val.append(self.values[t_start:t_end, env_idx])        # (seq_len,)

            seqs_ep.append(self.dones[t_start:t_end, env_idx])

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
        padded_val  = pad_sequence(seqs_val,  batch_first=True)  # (n_seq, max_len)
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
            flat(padded_val),
        )


class RecurrentPPOAgent:
    """
    PPO agent that uses LSTMActorCritic and RecurrentRolloutBuffer.
    Exposes the same learn(total_timesteps) interface as PPOAgent.
    """

    def __init__(self, envs, actor_critic, device,
                 lr_start=config.LR, lr_end=config.LR, lr_decay_start=1.0,
                 clip_range=config.BASELINE_CLIP_RANGE,
                 ent_coef_start=config.BASELINE_ENT_COEF,
                 ent_coef_end=0.0,
                 vf_coef=0.5):
        self.envs         = envs
        self.actor_critic = actor_critic
        self.device       = device

        # Hyperparameters (LSTM variants from config)
        self.lr_start     = lr_start
        self.lr_end       = lr_end
        self.lr_decay_start = lr_decay_start
        self.num_steps    = config.LSTM_N_STEPS
        self.batch_size   = config.LSTM_BATCH_SIZE
        self.n_epochs     = config.LSTM_N_EPOCHS
        self.gamma        = config.GAMMA
        self.clip_range     = clip_range
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end   = ent_coef_end
        self.ent_coef       = ent_coef_start
        self.vf_coef        = vf_coef

        self.n_critic_epochs  = config.LSTM_N_CRITIC_EPOCHS
        self.actor_optimizer  = optim.Adam(actor_critic.actor_parameters(),
                                           lr=self.lr_start, eps=1e-5)
        self.critic_optimizer = optim.Adam(actor_critic.critic_parameters(),
                                           lr=config.LSTM_CRITIC_LR, eps=1e-5)

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

    def learn(self, total_timesteps, on_episode=None, on_checkpoint=None, on_update=None):
        """
        Main training loop — same signature as PPOAgent.learn().

        Args:
            total_timesteps: total env steps to train for
            on_episode:      optional callable(ep_reward, global_step, ent_coef, llm_belief_arr)
            on_checkpoint:   optional callable(actor_critic, global_step)
            on_update:       optional callable(losses_dict, global_step)
        """
        global_step = 0
        _last_ckpt = 0

        raw_obs = self.envs.reset()
        next_obs = flatten_obs(raw_obs, self.device)
        next_done = torch.zeros(config.N_ENVS, device=self.device)

        # LSTM state carried across rollout windows
        self._last_hidden = None

        num_updates = total_timesteps // (self.num_steps * config.N_ENVS)

        # Episode tracking
        ep_rewards      = torch.zeros(config.N_ENVS, device=self.device)
        ep_norm_rewards = torch.zeros(config.N_ENVS, device=self.device)
        ep_llm_accs  = [[] for _ in range(config.N_ENVS)]

        for update in range(1, num_updates + 1):
            # --- Rollout collection ---
            for step in range(self.num_steps):
                global_step += config.N_ENVS

                with torch.no_grad():
                    action, logprob, value, new_hidden = self.actor_critic.get_action_and_value(
                        next_obs, self._last_hidden, next_done
                    )

                raw_new_obs, rewards, dones, infos = self.envs.step(action.cpu().numpy())

                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                dones_tensor   = torch.tensor(dones,   dtype=torch.float32, device=self.device)

                self.buffer.add(
                    next_obs,
                    action,
                    logprob,
                    rewards_tensor,
                    value.flatten(),
                    next_done,          # done from previous step — marks episode boundaries
                    self._last_hidden,  # hidden state at start of this step
                )

                # Track episode rewards (raw) and llm beliefs
                # VecNormalize stores pre-normalization rewards in .old_reward
                _old_r = getattr(self.envs, "old_reward", None)
                raw_rewards = _old_r if _old_r is not None else rewards
                ep_rewards      += torch.tensor(raw_rewards, dtype=torch.float32, device=self.device)
                ep_norm_rewards += rewards_tensor
                for i in range(config.N_ENVS):
                    ep_llm_accs[i].append(raw_new_obs["llm_belief"][i])

                for i, done in enumerate(dones):
                    if done:
                        llm_arr = np.array(ep_llm_accs[i]) if ep_llm_accs[i] else None
                        if on_episode:
                            on_episode(float(ep_rewards[i]), float(ep_norm_rewards[i]), global_step, self.ent_coef, llm_arr)
                        ep_rewards[i] = 0.0
                        ep_norm_rewards[i] = 0.0
                        ep_llm_accs[i] = []

                # Advance state
                self._last_hidden = new_hidden
                next_obs  = flatten_obs(raw_new_obs, self.device)
                next_done = dones_tensor

            # --- Bootstrap value for GAE ---
            with torch.no_grad():
                _, _, next_value, _ = self.actor_critic.get_action_and_value(
                    next_obs, self._last_hidden, next_done
                )

            self.buffer.compute_returns_and_advantages(
                next_value.flatten(), next_done, gamma=self.gamma
            )

            # Decay entropy coefficient and update LR
            progress = (update - 1) / num_updates
            self.ent_coef = self.ent_coef_end + (self.ent_coef_start - self.ent_coef_end) * (1.0 - progress)
            self._update_lr(progress)

            losses = self._update_policy()
            if on_update is not None:
                on_update(losses, global_step)

            # Reset buffer step counter (carry hidden state across windows)
            self.buffer.step = 0

            if on_checkpoint and global_step - _last_ckpt >= config.CHECKPOINT_FREQ:
                on_checkpoint(self.actor_critic, global_step)
                _last_ckpt = global_step

    def _update_lr(self, progress):
        flat_end = 1.0 - self.lr_decay_start
        if progress < flat_end:
            lr = self.lr_start
        else:
            remaining = 1.0 - progress
            lr = self.lr_end + (self.lr_start - self.lr_end) * (remaining / self.lr_decay_start)
        for pg in self.actor_optimizer.param_groups:
            pg["lr"] = lr

    def _update_policy(self):
        """
        Runs separate actor/critic update loops.
        Actor: n_epochs (PPO trust-region constrained).
        Critic: n_critic_epochs (pure regression, no constraint).
        """
        policy_loss_vals, value_loss_vals, entropy_loss_vals = [], [], []

        # --- Actor loop ---
        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_generator(self.batch_size):
                (b_obs, b_actions, b_old_logprobs, b_advantages, b_returns,
                 b_lstm_states, b_ep_starts, b_mask, b_old_values) = batch

                b_advantages = (b_advantages - b_advantages[b_mask].mean()) / (b_advantages[b_mask].std() + 1e-8)

                new_logprobs, entropy = self.actor_critic.evaluate_actor(
                    b_obs, b_actions, b_lstm_states, b_ep_starts
                )

                logratio    = new_logprobs - b_old_logprobs
                ratio       = logratio.exp()
                surr1       = ratio * b_advantages
                surr2       = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * b_advantages
                policy_loss = -torch.min(surr1, surr2)[b_mask].mean()
                entropy_loss = -entropy[b_mask].mean()
                loss = policy_loss + self.ent_coef * entropy_loss

                self.actor_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.actor_parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                policy_loss_vals.append(policy_loss.item())
                entropy_loss_vals.append(entropy_loss.item())

        # --- Critic loop ---
        for epoch in range(self.n_critic_epochs):
            for batch in self.buffer.get_generator(self.batch_size):
                (b_obs, b_actions, b_old_logprobs, b_advantages, b_returns,
                 b_lstm_states, b_ep_starts, b_mask, b_old_values) = batch

                new_values = self.actor_critic.evaluate_critic(
                    b_obs, b_lstm_states, b_ep_starts
                ).squeeze(-1)
                value_loss = ((new_values - b_returns) ** 2)[b_mask].mean()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.critic_parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                value_loss_vals.append(value_loss.item())

        # EV on buffer values (pre-update, as before)
        with torch.no_grad():
            y_true   = self.buffer.returns.flatten()
            y_pred   = self.buffer.values.flatten()
            var_y    = torch.var(y_true)
            expl_var = (1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)).item()

        return {
            "policy_loss":        float(np.mean(policy_loss_vals)),
            "value_loss":         float(np.mean(value_loss_vals)),
            "entropy_loss":       float(np.mean(entropy_loss_vals)),
            "explained_variance": expl_var,
        }

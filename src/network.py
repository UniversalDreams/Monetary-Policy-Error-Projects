import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization method
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    """
    MLP policy network for discrete action space
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # actor network (pi)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # output layer gets 0.01 std to ensure uniform initial exploration
            layer_init(nn.Linear(64, act_dim), std=0.01)
        )

        # critic network (vf)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # output layer gets 1.0 std
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def get_action_and_value(self, obs, hidden=None, episode_starts=None):
        """
        Called during rollout collection. hidden and episode_starts ignored for MLP.
        Returns 4-tuple (action, logprob, value, hidden=None) for unified interface.
        """
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        action = probs.sample()

        return action, probs.log_prob(action), self.critic(obs), None

    def evaluate_actions(self, obs, action, hidden=None, episode_starts=None):
        """
        Called during the PPO update epoch. hidden and episode_starts ignored for MLP.
        """
        logits = self.actor(obs)
        probs = Categorical(logits=logits)

        return probs.log_prob(action), probs.entropy(), self.critic(obs)


class LSTMActorCritic(nn.Module):
    """
    Recurrent policy network: shared MLP feature extractor → separate LSTM per actor/critic → heads.
    Mirrors sb3_contrib RecurrentPPO architecture.
    """
    def __init__(self, obs_dim, act_dim, lstm_hidden_size=64, n_lstm_layers=1):
        super().__init__()

        # Separate feature extractors for actor and critic (prevents value loss
        # from corrupting actor features via a shared encoder)
        self.features_pi = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.features_vf = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        # Separate LSTMs for actor and critic (batch_first=False: input is (T, N, input_size))
        self.lstm_actor  = nn.LSTM(64, lstm_hidden_size, num_layers=n_lstm_layers, batch_first=False)
        self.lstm_critic = nn.LSTM(64, lstm_hidden_size, num_layers=n_lstm_layers, batch_first=False)

        # Output heads
        self.actor_head  = layer_init(nn.Linear(lstm_hidden_size, act_dim), std=0.01)
        self.critic_head = layer_init(nn.Linear(lstm_hidden_size, 1),       std=1.0)

    @staticmethod
    def _process_sequence(features, h, c, episode_starts, lstm):
        """
        Run LSTM over a sequence, zeroing hidden state at episode boundaries.

        Args:
            features:       (n_seq * T, input_size) — flattened padded sequences
            h, c:           (n_layers, n_seq, hidden_size) — initial states
            episode_starts: (n_seq * T,) — 1.0 at sequence/episode starts, 0.0 otherwise
            lstm:           nn.LSTM module

        Returns:
            out:            (n_seq * T, hidden_size)
            (h_new, c_new): updated hidden states
        """
        n_seq = h.shape[1]
        # Reshape to (T, n_seq, input_size)
        features_seq = features.reshape(n_seq, -1, lstm.input_size).permute(1, 0, 2)
        episode_starts = episode_starts.reshape(n_seq, -1).permute(1, 0)  # (T, n_seq)

        if torch.all(episode_starts == 0.0):
            # Fast path: no resets needed in this batch
            out, (h_new, c_new) = lstm(features_seq, (h, c))
            out = out.permute(1, 0, 2).flatten(0, 1)  # (n_seq * T, hidden_size)
            return out, (h_new, c_new)

        # Step-by-step to zero states at episode boundaries
        outputs = []
        for feat_t, ep_start_t in zip(features_seq, episode_starts):
            # ep_start_t: (n_seq,)  — 1 means reset this env's hidden state
            mask = (1.0 - ep_start_t).view(1, n_seq, 1)
            out_t, (h, c) = lstm(feat_t.unsqueeze(0), (h * mask, c * mask))
            outputs.append(out_t)

        out = torch.cat(outputs, dim=0).permute(1, 0, 2).flatten(0, 1)  # (n_seq * T, hidden_size)
        return out, (h, c)

    def _unpack_or_init(self, hidden, n_envs):
        """Return (h_pi, c_pi, h_vf, c_vf) from hidden tuple, or zeros if hidden is None."""
        if hidden is None:
            device = next(self.parameters()).device
            zeros = lambda: torch.zeros(
                self.lstm_actor.num_layers, n_envs, self.lstm_actor.hidden_size, device=device
            )
            return zeros(), zeros(), zeros(), zeros()
        return hidden  # (h_pi, c_pi, h_vf, c_vf)

    def get_action_and_value(self, obs, hidden=None, episode_starts=None):
        """
        Called during rollout (single step per env).

        Args:
            obs:            (n_envs, obs_dim)
            hidden:         (h_pi, c_pi, h_vf, c_vf) or None → init zeros
            episode_starts: (n_envs,) float — 1.0 if this step starts a new episode

        Returns: (action, logprob, value, new_hidden)
        """
        n_envs = obs.shape[0]
        h_pi, c_pi, h_vf, c_vf = self._unpack_or_init(hidden, n_envs)
        if episode_starts is None:
            episode_starts = torch.zeros(n_envs, device=obs.device)

        feat_pi = self.features_pi(obs)
        feat_vf = self.features_vf(obs)

        latent_pi, (h_pi_new, c_pi_new) = self._process_sequence(
            feat_pi, h_pi, c_pi, episode_starts, self.lstm_actor
        )
        latent_vf, (h_vf_new, c_vf_new) = self._process_sequence(
            feat_vf, h_vf, c_vf, episode_starts, self.lstm_critic
        )

        logits = self.actor_head(latent_pi)
        probs  = Categorical(logits=logits)
        action = probs.sample()
        value  = self.critic_head(latent_vf)

        new_hidden = (h_pi_new, c_pi_new, h_vf_new, c_vf_new)
        return action, probs.log_prob(action), value, new_hidden

    def actor_parameters(self):
        return (list(self.features_pi.parameters()) +
                list(self.lstm_actor.parameters()) +
                list(self.actor_head.parameters()))

    def critic_parameters(self):
        return (list(self.features_vf.parameters()) +
                list(self.lstm_critic.parameters()) +
                list(self.critic_head.parameters()))

    def evaluate_actor(self, obs, action, hidden=None, episode_starts=None):
        n_seq = hidden[0].shape[1]
        h_pi, c_pi = hidden[0], hidden[1]
        if episode_starts is None:
            episode_starts = torch.zeros(obs.shape[0], device=obs.device)
        feat_pi = self.features_pi(obs)
        latent_pi, _ = self._process_sequence(feat_pi, h_pi, c_pi, episode_starts, self.lstm_actor)
        logits = self.actor_head(latent_pi)
        probs = Categorical(logits=logits)
        return probs.log_prob(action), probs.entropy()

    def evaluate_critic(self, obs, hidden=None, episode_starts=None):
        h_vf, c_vf = hidden[2], hidden[3]
        if episode_starts is None:
            episode_starts = torch.zeros(obs.shape[0], device=obs.device)
        feat_vf = self.features_vf(obs)
        latent_vf, _ = self._process_sequence(feat_vf, h_vf, c_vf, episode_starts, self.lstm_critic)
        return self.critic_head(latent_vf)

    def evaluate_actions(self, obs, action, hidden=None, episode_starts=None):
        """
        Called during the PPO update epoch with padded sequence batches.

        Args:
            obs:            (n_seq * max_len, obs_dim)
            action:         (n_seq * max_len,)
            hidden:         (h_pi, c_pi, h_vf, c_vf), each (n_layers, n_seq, hidden_size)
            episode_starts: (n_seq * max_len,) — 1.0 at sequence starts

        Returns: (logprob, entropy, value)
        """
        n_seq = hidden[0].shape[1]
        h_pi, c_pi, h_vf, c_vf = hidden
        if episode_starts is None:
            episode_starts = torch.zeros(obs.shape[0], device=obs.device)

        feat_pi = self.features_pi(obs)
        feat_vf = self.features_vf(obs)

        latent_pi, _ = self._process_sequence(feat_pi, h_pi, c_pi, episode_starts, self.lstm_actor)
        latent_vf, _ = self._process_sequence(feat_vf, h_vf, c_vf, episode_starts, self.lstm_critic)

        logits = self.actor_head(latent_pi)
        probs  = Categorical(logits=logits)
        return probs.log_prob(action), probs.entropy(), self.critic_head(latent_vf)
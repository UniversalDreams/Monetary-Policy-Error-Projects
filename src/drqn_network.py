import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal Init"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class DuelingDRQN(nn.Module):
    """
    Deep Recurrent Q-Network
    """
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        self.act_dim = act_dim
        self.hidden_size = hidden_size

        # feature extractor
        self.feature_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.SiLU()
        )

        # recurrent core
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # value stream: how good is the economic state
        self.value_stream = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0)
        )

        # advantage stream: how good is the interest rate action relative to others
        self.advantage_stream = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.SiLU(),
            layer_init(nn.Linear(hidden_size, act_dim), std=0.01)
        )

    def forward(self, x, hidden_state=None):
        batch_size, seq_len, _ = x.size()

        # flatten for feature extractor
        x_flat = x.view(batch_size * seq_len, -1)
        features = self.feature_net(x_flat)

        # reshape for lstm
        features = features.view(batch_size, seq_len, -1)
        lstm_out, hidden_state = self.lstm(features, hidden_state)

        # flatten for dueling heads
        lstm_flat = lstm_out.contiguous().view(batch_size * seq_len, -1)

        # calculate value and advantage
        values = self.value_stream(lstm_flat)
        advantages = self.advantage_stream(lstm_flat)

        # combine using dueling formula: Q = V + (A - mean(A))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        q_values = q_values.view(batch_size, seq_len, self.act_dim)
        return q_values, hidden_state


    def get_initial_hidden_state(self, batch_size, device):
        """Get init hidden states for the start of an episode"""
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return h_0, c_0
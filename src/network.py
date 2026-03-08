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

    def get_action_and_value(self, obs):
        """
        Called during rollout collection
        """
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        action = probs.sample()

        return action, probs.log_prob(action), self.critic(obs)

    def evaluate_actions(self, obs, action):
        """
        Called during the PPO update epoch
        """
        logits = self.actor(obs)
        probs = Categorical(logits=logits)

        return probs.log_prob(action), probs.entropy(), self.critic(obs)
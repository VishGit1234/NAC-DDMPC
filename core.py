import numpy as np

import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        # q = self.q(torch.cat([obs, act], dim=-1))
        q = self.q(obs)
        # return torch.squeeze(q, -1) # Critical to ensure q has right shape.
        return q

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # build policy and value functions
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False, alpha=0.2):
        with torch.no_grad():
            q = self.q(obs)
            v = alpha*torch.log(torch.exp(q/alpha).sum(-1))
            policy = torch.exp((q - v)/alpha)
            a = policy.argmax(-1)

            # a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
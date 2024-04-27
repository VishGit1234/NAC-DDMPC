import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
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
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        # q = self.q(obs)
        # return torch.squeeze(q, -1) # Critical to ensure q has right shape.
        return q

class MLPActionSampler(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # Turns states into a distribution (parameterized by mean and std dev) 
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, with_logprob=False):
        # Get mean and standard deviation
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Sample from this distribution using rparam trick (adds stochasticity)
        dist = Normal(mu, std)
        a = dist.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = dist.log_prob(a).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - a - F.softplus(-2*a))).sum(axis=1)
        else:
            logp_pi = None

        # # sample from standard normal distribution (array of length batchsize)
        # batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
        # nml = torch.normal(torch.zeros(batch_size), torch.ones(batch_size))
        # nml.requires_grad = False
        
        # # feed both observation and normal dist. sample into network
        # a = self.a(torch.cat([obs, nml], dim=-1))
        # bound action values between -1 and 1
        a = torch.tanh(a)

        return a, logp_pi

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # build q function and action sampling function
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.a = MLPActionSampler(obs_dim, act_dim, hidden_sizes, nn.ReLU)

    def act(self, obs, deterministic=False, alpha=0.2):
        with torch.no_grad():
            # sample an action
            a, _ = self.a(obs)
            return a.numpy()
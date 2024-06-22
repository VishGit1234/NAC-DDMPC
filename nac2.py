import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from math import prod
import itertools

def combined_shape(s1, s2):

    return s1 + s2


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim: int, act_dim: int, size : int):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum(np.prod(p.shape) for p in module.parameters())


class MLPQfunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + hidden_sizes + [1], activation)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))


LOG_STD_MAX = 3
LOG_STD_MIN = -15

class MLPActionSampler(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + hidden_sizes, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, is_determinstic = False, with_logprob=True):
        net_out = self.net(obs)

        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std,LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)


        if(is_determinstic):
            return mu

        a = dist.rsample()
        if with_logprob:
            logp_a = dist.log_prob(a).sum(axis=-1)
            logp_a -= (2*(np.log(2) - a - F.softplus(-2*a))).sum()
            logp_a = logp_a.squeeze()
        else:
            logp_a = None

        return torch.tanh(a), logp_a



class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden_sizes=[32,32], activation=nn.ReLU):
        super().__init__()
        self.act_low = act_low
        self.act_high = act_high

        # build q function and action sampling function
        self.q = MLPQfunction(obs_dim, act_dim, hidden_sizes, activation)
        self.pi = MLPActionSampler(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            # sample an action
            return self.pi(obs, is_determinstic = deterministic , with_logprob=False)
        

#action sampler should be identical to SAC
def loss_pi_0(batch, ac,ac_targ, alpha, gamma):

def loss_pi_1(batch, ac, ac_targ, alpha, gamma):
    o = batch['obs']
    pi, logp_pi = ac.pi(o)
    q_pi = ac.q(o, pi)

    # Entropy-regularized policy loss
    loss_pi = (alpha * logp_pi - q_pi).square().mean()

    # Useful info for logging
    pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

    return loss_pi, pi_info



def uniform_sample(shape,low,high):



    sample = torch.rand(combined_shape(shape, low.shape))
    sample = sample.mul(torch.tensor(high)- torch.tensor(low))
    sample = sample.add(torch.tensor(low))

    return sample



 #Expects flattened observations and actions
def update_loss_Jv(batch, ac, ac_targ ,alpha, gamma, q_optim, n_samples = 128, ):
    o, a, r, o2, d = batch['obs'], batch['act'], batch['rew'], batch['obs2'], batch['done']


    batch_size = o.shape[0]
    obs_dim = o.shape[1]
    act_dim  = a.shape[1]

    
    #compute o state value by sampling from action space
    o_samples = o.unsqueeze(1).repeat(1, n_samples,1)#shape should be (n_samples, batch_size, obs_dim)
    assert o_samples.shape == (batch_size, n_samples, obs_dim)
    a_samples = uniform_sample((batch_size, n_samples), ac.act_low, ac.act_high)#shape should be (batch_size, n_samples, act_dim)
    assert a_samples.shape == (batch_size, n_samples, act_dim)
    #compute sample q values, q(. , s)
    q1_samples = ac.q(o_samples, a_samples).squeeze() # q1 has shape (batch_size, n_samples)
    assert q1_samples.shape == (batch_size, n_samples)
    v1 = (alpha* torch.logsumexp(q1_samples/alpha, dim=1)).squeeze()#v1 has shape (batch_size)
    
    v1_samples = v1.unsqueeze(1).repeat(1, n_samples)
    #independent sample for q
    a_samples2 = uniform_sample((batch_size, n_samples), ac.act_low, ac.act_high)
    q1_samples = ac.q(o_samples, a_samples2).squeeze()

    pi_samples = torch.exp((q1_samples-v1_samples)/alpha)

    ent_sa = 0
    #assert ent_sa.shape == (batch_size, )
    #assert ent_sa > 0
    
    #Compute V hat 
    o2_samples = o2.unsqueeze(1).repeat(1, n_samples,1)#shape should be (n_samples, batch_size, obs_dim)
    #assert o2_samples.shape == (batch_size, n_samples, obs_dim)
    a2_samples = uniform_sample((batch_size, n_samples), ac.act_low, ac.act_high) #shape should be (n_samples, batch_size, act_dim)
    #assert a2_samples.shape == (batch_size, n_samples , act_dim)
    q2_samples = ac.q(o2_samples, a2_samples) # q1 has shape (batch_size)
    #assert q2.shape == (batch_size, n_samples,1)
    v2 = alpha* torch.logsumexp(q2_samples/alpha, dim=1).squeeze() #v1 has shape (batch_size)
    #assert v2.shape == (batch_size,)

    #a2, logp_a2 = ac_targ.pi(o2)
    v_hat = r + gamma*v2  + alpha*ent_sa


    J_v = 0.5*(v1-v_hat).square().mean()
    J_v.backward()
    q_optim.step()
    q_optim.zero_grad()
    return J_v


def update_loss_Jpg(batch, ac, ac_targ, alpha, gamma, q_optim, n_samples = 128, ):
    o, a, r, o2, d = batch['obs'], batch['act'], batch['rew'], batch['obs2'], batch['done']
    batch_size = o.shape[0]
    obs_dim = o.shape[1]
    act_dim  = a.shape[1]
    #policy gradient loss
    o_samples = o.unsqueeze(1).repeat(1, n_samples,1)#shape should be (n_samples, batch_size, obs_dim)
    assert o_samples.shape == (batch_size, n_samples, obs_dim)
    a_samples = uniform_sample((batch_size, n_samples), ac.act_low, ac.act_high)#shape should be (batch_size, n_samples, act_dim)
    assert a_samples.shape == (batch_size, n_samples, act_dim)
    #compute sample q values, q(. , s)
    q1_samples = ac.q(o_samples, a_samples).squeeze() # q1 has shape (batch_size, n_samples)
    assert q1_samples.shape == (batch_size, n_samples)
    v1 = (alpha* torch.logsumexp(q1_samples/alpha, dim=1)).squeeze()#v1 has shape (batch_size)
    #loss for jpg
    q_sa = ac.q(o,a).squeeze()
    
    J_pg = (q_sa - v1).sum()
    #after this step, ac.q should have gradients mean(grad_theta(q_sa-v1))
    J_pg.backward()


    with torch.no_grad():
        ##COMPUTE Q HAT USING TARGET NETWORK
        o2_samples = o.unsqueeze(1).repeat(1, n_samples,1)
        a_samples = uniform_sample((batch_size, n_samples), ac.act_low, ac.act_high)#shape should be (batch_size, n_samples, act_dim)
        q2_samples = ac_targ.q(o2_samples, a_samples)
        v2 = alpha*torch.logsumexp(q2_samples/alpha, dim=1).squeeze()
        q_hat = r + gamma*v2
        q_minus_qhat = q_sa - q_hat

        for p in ac.q.parameters():
            #print(p.grad)
            g1 = p.grad.unsqueeze(0).repeat((batch_size, ) + tuple(1 for i in p.grad.shape) )
            #g1 has shape (*p.grad.shape, batch_size)
            assert g1.shape ==  combined_shape((batch_size,), p.grad.shape)
            
            coif = q_minus_qhat.reshape((batch_size,) + tuple(1 for i in p.grad.shape))


            p.grad = (g1*coif).sum(axis=0)
    q_optim.step()
    q_optim.zero_grad()
    return q_minus_qhat.square().mean()


def test_agent(ac, test_env):
    f = 0.0
    for j in range(3):
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        o = o[0]
        while not(d or (ep_len == 200)):
            # Take deterministic actions at test time
            o, r, d, _, _ = test_env.step(ac.act(torch.tensor(o, dtype=torch.float32), deterministic=True).numpy())
            ep_ret += r
            ep_len += 1
        # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        f += 1.0 if d else 0
    return ep_ret, ep_len, f/3


def sample_demos(env_fn, steps):
    env = env_fn()
    env.metadata["render_fps"]= 0
    obs_dim = prod(i for i in env.observation_space.shape)
    act_dim = prod(i for i in env.action_space.shape)
    demo_buffer = ReplayBuffer(obs_dim, act_dim, steps)


    o, _ = env.reset()
    for t in range(steps):
        a = env.action_space.sample()
        o2,r,d,_,_ = env.step(a)
        demo_buffer.store(o,a,r,o2,d)
        o = o2
        if(d):
            o, _ = env.reset()

    return demo_buffer


def nac(env_fn, steps,  start_steps, update_every, alpha=0.4, gamma=0.99, lr=1e-5, realtime=False, actor_critic = MLPActorCritic, buffer_size=10000, batch_size=64, demo_steps = 2000, polyak = 0, report_every = 2000, sac_ac= None):
    torch.manual_seed(0)
    np.random.seed(0)

    env = env_fn()



    obs_dim = int( prod(i for i in env.observation_space.shape))
    act_dim = int(prod(i for i in env.action_space.shape))

    if not realtime:
        env.metadata["render_fps"]= 0

    ac = actor_critic(obs_dim, act_dim, env.action_space.low, env.action_space.high)
    ac_targ = actor_critic(obs_dim, act_dim, env.action_space.low, env.action_space.high)
    ac_targ.requires_grad_ = False


    demo_buffer = sample_demos(env_fn, demo_steps)
    replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_size)
    
    q_optim = torch.optim.Adam(ac.q.parameters(), lr)
    pi_optim = torch.optim.Adam(ac.pi.parameters(), lr)

    ep_ret = 0

    print("finished sampling")

    o, _ = env.reset()

    o = torch.tensor(o, dtype=torch.float32)
    for t in range(steps):

        action_batch=32

        a, _ = ac.act(o)
        if t%500 == 0:
            q_nac = ac.q(o,a)
            q_sac = sac_ac.q1(o,a)

            s_act = sac_ac.act(o)
            n_act = ac.act(o)

            #print(o,  q_samples1, s_act)
            
            print(f"{o}, sac act:{s_act} nac act:{n_act}, nac q {q_nac}, sac q {q_sac}")      



        o2, r, d, _, _ = env.step(a.cpu())
        replay_buffer.store(o.flatten(),a.flatten(),r,o2,d)
        o = o2
        o = torch.tensor(o, dtype=torch.float32)
        
        


        if(d):
            o, _ = env.reset()

        if(t < start_steps):
            batch = demo_buffer.sample_batch(batch_size)
        else:
            batch = replay_buffer.sample_batch(batch_size)

        #update fast response
        q_optim.zero_grad()
        lq = update_loss_Jv(batch, ac, ac_targ ,alpha, gamma, q_optim)
        lq = update_loss_Jpg(batch, ac, ac_targ, alpha, gamma, q_optim)
        pi_optim.zero_grad()
        lpi, _ = loss_pi_0(batch, ac, ac_targ, alpha, gamma)
        lpi.backward()
        pi_optim.step()


  
        #update slow response
        if t% update_every == 0:

            
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                    
                    #assert torch.isclose(p_targ, p).all()

                    #print(p_targ)

        
        if t% report_every == 0:
            ep_ret, ep_len, d = test_agent(ac, env)
            lpi = 0
            print(f"epoch: {t//report_every} epoch length: {report_every}, ep_len: {ep_len}, ep_ret: {ep_ret} finished: {d}, loss q: {lq}, loss pi: {lpi}")


    return ac

import gymnasium as gym
from sac import sac
import sac_core

if __name__ == '__main__':
    torch.set_default_device('cpu')
    

    ac_sac = sac(num_envs=8, env_fn=lambda : gym.make("Pendulum-v1", render_mode=None), actor_critic=sac_core.MLPActorCritic, realtime=False, start_steps=10000, replay_size=1000000, update_every=50, update_after=10000, batch_size=256, steps_per_epoch=5000, max_ep_len=200, num_test_episodes=1, lr=1e-3, epochs=1)
    ac_nac = nac(env_fn=lambda : gym.make("Pendulum-v1", render_mode=None), steps=100000, start_steps=8000,update_every=50, 
        alpha=0.2, gamma=0.99, lr=1e-3, realtime=False, actor_critic = MLPActorCritic, buffer_size=100000, batch_size=100, demo_steps = 10000, polyak = 0.995, report_every = 2000, sac_ac=ac_sac) 
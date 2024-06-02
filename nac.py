from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import core
import math
# import gym
import time
from timeit import default_timer
from stable_baselines3.common.vec_env import SubprocVecEnv

from kernel import adaptive_isotropic_gaussian_kernel_stable as adaptive_isotropic_gaussian_kernel_stable
# import spinup.algos.pytorch.sac.core as core
# from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, 1), dtype=np.float32)
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



def nac(num_envs, env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-5, alpha=0.4, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, realtime=False):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        realtime (bool): Wheather to run the simulation in realtime or not, (set to false for faster training)

    """

    #logger = EpochLogger(**logger_kwargs)
    #logger.save_config(locals())
    

    torch.manual_seed(seed)
    np.random.seed(seed)

    test_env = env_fn()
    envs = SubprocVecEnv([env_fn for _ in range(num_envs)])
    obs_dim = envs.observation_space.shape
    act_dim = envs.action_space.shape[0]

    if not realtime:
        envs.set_attr('render_fps', 0)#['render_fps'] = 0


    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_high = envs.action_space.high[0]
    act_low = envs.action_space.low[0]

    # Create actor-critic module and target networks
    ac = actor_critic(envs.observation_space.shape, envs.action_space.shape, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # torch.autograd.set_detect_anomaly(True)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both networks
    # q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    demonstrations = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=start_steps)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # Calculate value for each state in mini-batch using importance sampling derived formula
        n_samples = batch_size # Number of samples for sample mean
        # duplicate o_samples based on number of samples
        o_samples = torch.unsqueeze(o, 1).repeat(1, n_samples, *([1]*len(o.shape[1:])))
        o_samples = o_samples.reshape(o_samples.shape[0] * o_samples.shape[1], *o_samples.shape[2:])
        a_samples, _ = ac.a(o_samples)
        q_samples = ac.q(o_samples, a_samples)
        # Reshape to B x num_samples
        q_samples = q_samples.reshape(batch_size, n_samples)

        # Compute values of current states (I think dividing by q_ai(ai) can be ignored)
        # alpha*(LSE(q/a) - log(N))
        # Using torch operations for differentiability
        v = torch.mul(torch.sub(torch.logsumexp(q_samples/alpha, dim=1), math.log(n_samples)), alpha)

        # Get q-value estimates for all state-action pairs in mini-batch
        q = ac.q(o, a)

        # Compute Q target and V target
        with torch.no_grad():
            o2_samples = torch.unsqueeze(o2, 1).repeat(1, n_samples, *([1]*len(o2.shape[1:])))
            o2_samples = o2_samples.reshape(o2_samples.shape[0] * o2_samples.shape[1], *o2_samples.shape[2:])
            a_samples2, logps = ac.a(o2_samples, True)
            logps = logps.reshape(batch_size, n_samples)
            q_samples2 = ac.q(o2_samples, a_samples2)
            q_samples2 = q_samples2.reshape(batch_size, n_samples)

            # Compute value of next state
            v2 = torch.logsumexp(q_samples2/alpha, dim=1)
            v2 -= math.log(n_samples)
            v2 *= alpha

            # Compute targets
            q_targ = torch.unsqueeze(r, dim=-1) + gamma*v2
            v_targ = q_samples2.mean(dim=1) + alpha*torch.unsqueeze(logps.mean(dim=1), dim=-1)

        # Calculate gradients
        policy_grad = ((q - v)*(q - q_targ)).mean(dim=-1)
        v_grad = torch.div(torch.square(v - v_targ), 2).mean(dim=-1)

        loss_q = -(policy_grad + v_grad)
        return torch.mean(loss_q)

    # Set up function for computing NAC actor loss
    def compute_loss_a(data):
        o = data['obs']
        
        # Check values of observation tensor
       # print("Observation tensor:", o)

        n_samples = batch_size
        n_fixed_actions = n_samples // 2
        fixed_actions = []
        o_samples = torch.unsqueeze(o, 1).repeat(1, n_fixed_actions, *([1]*len(o.shape[1:])))
        o_samples = o_samples.reshape(o_samples.shape[0] * o_samples.shape[1], *o_samples.shape[2:])
        fixed_actions, _ = ac.a(o_samples)
        fixed_actions = fixed_actions.reshape(batch_size, n_fixed_actions).unsqueeze(-1)

        # Check values of fixed actions tensor
       # print("Fixed actions tensor:", fixed_actions)

        n_updated_actions = n_samples - n_fixed_actions
        o_samples2 = torch.unsqueeze(o, 1).repeat(1, n_updated_actions, *([1]*len(o.shape[1:])))
        o_samples2 = o_samples2.reshape(o_samples2.shape[0] * o_samples2.shape[1], *o_samples2.shape[2:])
        updated_actions, _ = ac.a(o_samples2)
        updated_actions = updated_actions.reshape(batch_size, n_updated_actions).unsqueeze(-1)

        # Check values of updated actions tensor
       # print("Updated actions tensor:", updated_actions)

        # flatten first 2 dims to input into network
        repeat_sizes = [1] * len(o.shape)
        repeat_sizes[0] *= n_fixed_actions
        svgd_target_values = ac.q(o.repeat(repeat_sizes), torch.flatten(fixed_actions, end_dim=1))
        svgd_target_values = torch.reshape(svgd_target_values, (o.shape[0], n_fixed_actions, act_dim))
        squash_correction = torch.sum(torch.log(torch.add(torch.neg(torch.square(fixed_actions)), 1 + 1e-6)), dim=-1)
        squash_correction = torch.unsqueeze(squash_correction, dim=-1)
        log_p = svgd_target_values + squash_correction
        
        # Check log_p tensor
       # print("log_p tensor:", log_p)

        grad_log_p = torch.autograd.grad(log_p, fixed_actions, torch.ones(batch_size, n_fixed_actions, 1), retain_graph=True, allow_unused=True)[0]
        grad_log_p = torch.unsqueeze(grad_log_p, dim=2)
        grad_log_p.requires_grad = False




        kernel_dict = adaptive_isotropic_gaussian_kernel_stable(xs=fixed_actions, ys=updated_actions)

        # Kernel function in Equation 13:
        kappa = torch.unsqueeze(kernel_dict["output"], dim=3)



        if torch.isnan(kappa).any():
            raise ValueError("NaN detected in action kappa")
        
        if torch.isnan(grad_log_p).any():
            raise ValueError("NaN detected in grad log p")
        
        if torch.isnan(kernel_dict["gradient"]).any():
            raise ValueError("NaN detected in kernel gradient")

        # Stein Variational Gradient in Equation 13:
        action_gradients = torch.mean(
            kappa * grad_log_p + kernel_dict["gradient"], dim=1)

        # Check action gradients tensor
        # print("Action gradients tensor before backward:", action_gradients)
        # print("Contains NaN:", torch.isnan(action_gradients).any().item())

        # This computes our gradients
        if torch.isnan(action_gradients).any():
            raise ValueError("NaN detected in action gradients")

        updated_actions.backward(action_gradients)

        # Calculate surrogate loss
        surrogate_loss = 0
        layers = 0
        for w in ac.a.parameters():
            surrogate_loss += torch.mean(w*w.grad.detach())
            layers += 1
        surrogate_loss /= layers

        ac.a.zero_grad()

        return surrogate_loss

    # Set up optimizers for action-sampler and q-function
    q_optimizer = Adam(ac.q.parameters(), lr=lr)
    a_optimizer = Adam(ac.a.parameters(), lr=lr)

    # Set up model saving
    #logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q and A
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False
        loss_a = compute_loss_a(data)
        loss_a.backward()
        q_optimizer.step()
        a_optimizer.step()
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        # logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        # for p in q_params:
        #     p.requires_grad = False

        # Next run one gradient descent step for pi.
        # pi_optimizer.zero_grad()
        # loss_pi, pi_info = compute_loss_pi(data)
        # loss_pi.backward()
        # pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        # for p in q_params:
        #     p.requires_grad = True

        # Record things
        # logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac_targ.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic, alpha)

    def test_agent():
        d = False
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            o = o[0]
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        return ep_ret, ep_len, d
            
    def train():
        # Prepare for interaction with environment
        total_steps = steps_per_epoch * epochs
        start_time = time.time()
        o = envs.reset()
        ep_ret, ep_len = np.zeros((num_envs,)), np.zeros((num_envs,))

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > start_steps:
                a = get_action(o)
            else:
                # Query expert demonstrator
                # raise NotImplementedError
                a = np.array([envs.action_space.sample() for _ in range(num_envs)])

            # Step the env
            o2, r, d, _ = envs.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            # d = False if ep_len==max_ep_len else d

            # Store experience to replay buffer or demonstrations
            if t > start_steps:
                [replay_buffer.store(_o, _a, _r, _o2, _d) for _o, _a, _r, _o2, _d in zip(o, a, r, o2, d)]
            else:
                # [demonstrations.store(_o, _a, _r, _o2, _d) for _o, _a, _r, _o2, _d in zip(o, a, r, o2, d)]
                [replay_buffer.store(_o, _a, _r, _o2, _d) for _o, _a, _r, _o2, _d in zip(o, a, r, o2, d)]
            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            for env_idx, _d in enumerate(d):
                if _d:
                    ep_len[env_idx] = 0
                    ep_ret[env_idx] = 0

            # Update handling
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    if t > start_steps:
                        batch = replay_buffer.sample_batch(batch_size)
                    else:
                        # batch = demonstrations.sample_batch(batch_size)
                        batch = replay_buffer.sample_batch(batch_size)
                    update(data=batch)

            # End of epoch handling
            if (t+1) % steps_per_epoch == 0:
                epoch = (t+1) // steps_per_epoch

                # Save model
                # if (epoch % save_freq == 0) or (epoch == epochs):
                    # logger.save_state({'env': env}, None)

                # Test the performance of the deterministic version of the agent.
                ep_ret, ep_len, d = test_agent()
                print(f"Epoch: {epoch}, Ep Len: {ep_len}, Ep Retu: {ep_ret}, Successful: {d}")
                ep_ret, ep_len = np.zeros((num_envs,)), np.zeros((num_envs,))
    
    # test_agent()
    train()
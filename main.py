from nac2 import nac
from sac import sac
import gymnasium as gym
import core
import sac_core
import torch

if __name__ == '__main__':
    torch.set_default_device('cuda')  
    nac(lambda : gym.make("Pendulum-v1", render_mode="human"), actor_critic=core.MLPActorCritic, realtime=False, start_steps=200, replay_size=100, update_every=10, update_after=200, batch_size=50, steps_per_epoch=5)

    #sac(num_envs=10, env_fn=lambda : gym.make("Pendulum-v1", render_mode='human'), actor_critic=sac_core.MLPActorCritic)
         #start_steps=200, replay_size=100, update_every=10, update_after=200, batch_size=50) 
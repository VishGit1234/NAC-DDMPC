from nac import nac
from sac import sac
import gymnasium as gym
import core
import sac_core
import torch

if __name__ == '__main__':
    torch.set_default_device('cuda')  
    nac(num_envs=2, env_fn=lambda : gym.make("Pendulum-v1"), actor_critic=core.MLPActorCritic, realtime=False, start_steps=10000, replay_size=1000000, update_every=50, update_after=10000, batch_size=256, steps_per_epoch=5000, max_ep_len=200, num_test_episodes=1, lr=1e-3, epochs=7)

    # sac(num_envs=2, env_fn=lambda : gym.make("Pendulum-v1"), actor_critic=sac_core.MLPActorCritic, realtime=False, start_steps=10, replay_size=5, update_every=1, update_after=20, batch_size=2, steps_per_epoch=50)

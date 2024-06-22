from nac2 import nac
from sac import sac
import gymnasium as gym
import core
import sac_core
import torch

if __name__ == '__main__':
    torch.set_default_device('cpu')
    #nac( env_fn=lambda : gym.make("MountainCarContinuous-v0", render_mode=None), actor_critic=core.MLPActorCritic, realtime=False, start_steps=10000, replay_size=1000000, update_every=50, update_after=10000, batch_size=256, steps_per_epoch=5000, max_ep_len=200, num_test_episodes=1, lr=1e-3, epochs=7)

    sac(num_envs=8, env_fn=lambda : gym.make("Pendulum-v1", render_mode=None), actor_critic=sac_core.MLPActorCritic, realtime=False, start_steps=10000, replay_size=1000000, update_every=50, update_after=10000, batch_size=256, steps_per_epoch=5000, max_ep_len=200, num_test_episodes=1, lr=1e-3, epochs=7)
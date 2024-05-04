from nac import nac
from sac import sac
import gymnasium as gym
import core
import sac_core

if __name__ == '__main__':
    nac(lambda : gym.make("MountainCarContinuous-v0", render_mode='human'), actor_critic=core.MLPActorCritic)
        # start_steps=200, replay_size=100, update_every=10, update_after=200, batch_size=50, steps_per_epoch=1000)

    sac(lambda : gym.make("MountainCarContinuous-v0", render_mode='human'), actor_critic=sac_core.MLPActorCritic,
         start_steps=200, replay_size=100, update_every=10, update_after=200, batch_size=50) 
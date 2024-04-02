from nac import nac
import gymnasium as gym
import core

if __name__ == '__main__':
    nac(lambda : gym.make("CartPole-v1", render_mode="human"), actor_critic=core.MLPActorCritic)
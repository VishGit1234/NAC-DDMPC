from nac import sac
import gymnasium as gym
import core

if __name__ == '__main__':
    sac(lambda : gym.make("CartPole-v1", render_mode="human"), actor_critic=core.MLPActorCritic)
import gym
import ctm2_envs
import time

env = gym.make("Distal-2-Tube-Reach-v0")
observation = env.reset()
for _ in range(1000):
    env.render()
    time.sleep(0.10)
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()

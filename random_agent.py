import argparse
import gym
import ctm2_envs
import ctr_envs
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="Exact-Ctr-3-Tube-Reach-v0")
    parser.add_argument('--render-type', type=str, default="human")
    parser.add_argument('--ros-flag', type=bool, default=True)

    args = parser.parse_args()

    env = gym.make(args.env, ros_flag=args.ros_flag, render_type=args.render_type)
    observation = env.reset()
    for _ in range(1000):
        env.render()
        time.sleep(0.10)
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
    env.close()

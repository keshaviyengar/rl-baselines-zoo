import argparse
import gym
import ctm2_envs
import ctr_envs
import time

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="Distal-2-Tube-Reach-v0")
    parser.add_argument('--render-type', type=str, default="")
    parser.add_argument('--ros-flag', type=bool, default=False)

    args = parser.parse_args()

    num_points = 1e5
    env = gym.make(args.env, ros_flag=args.ros_flag, render_type=args.render_type)
    dg_list = np.empty((3, int(num_points)))
    for i in range(int(num_points)):
        print(i)
        desired_goal = env.reset()['desired_goal']
        dg_list[:,i] = desired_goal

    env.close()
    plt.plot(dg_list[0,:], dg_list[2,:], 'o')
    plt.show()

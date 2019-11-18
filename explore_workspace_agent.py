import gym
import ctm2_envs
import rospy

import time
import sys
import os
import signal

import numpy as np

import subprocess, shlex

"""
Handler is used for ctrl-c during loop to exit gracefully
"""


class WorkspaceAgent:
    def __init__(self, max_ext_action=0.0001, max_rot_action=5, one_tube_limits=[360, 0.05],
                 two_tube_limits=[360, 0.10, 360, 0.05]):
        self.one_tube_limits = one_tube_limits
        self.two_tube_limits = two_tube_limits
        self.max_ext_action = max_ext_action
        self.max_rot_action = max_rot_action
        self.num_steps = None

        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signal, frame):
        sys.exit(0)

    def _init_one_tube(self):
        self.env = None
        self.env = gym.make("Distal-1-Tube-Reach-v0", ros_flag=True, render_type='human',
                            action_orientation_limit=30, action_length_limit=0.005,
                            initial_q_pos=[-1, 0, -0.05], n_substeps=1)
        self.action = np.zeros_like(self.env.action_space.low)
        self.ext_action_value = self.max_ext_action / self.env.n_substeps
        self.rot_action_value = self.max_rot_action / self.env.n_substeps
        self.num_steps = [int(self.one_tube_limits[0] / self.rot_action_value),
                          int(self.one_tube_limits[1] / self.ext_action_value)]
        print("1-tube num_steps: ", self.num_steps)

    def __init_two_tube(self):
        self.env = None
        self.env = gym.make("Distal-2-Tube-Reach-v0", ros_flag=True, render_type='human',
                            action_orientation_limit=30, action_length_limit=0.005,
                            initial_q_pos=[-1, 0, -0.10, -1, 0, -0.05], n_substeps=1)
        self.action = np.zeros_like(self.env.action_space.low)
        self.ext_action_value = self.max_ext_action / self.env.n_substeps
        self.rot_action_value = self.max_rot_action / self.env.n_substeps
        self.num_steps = [int(self.two_tube_limits[0] / self.rot_action_value),
                          int(self.two_tube_limits[1] / self.ext_action_value),
                          int(self.two_tube_limits[2] / self.rot_action_value),
                          int(self.two_tube_limits[3] / self.ext_action_value)]
        print("2-tube num_steps: ", self.num_steps)

    def ext_tube(self, tube_id, retract=False):
        ext_action = np.zeros_like(self.env.action_space.high)
        if retract:
            ext_action[tube_id + 1] = -self.ext_action_value
        else:
            ext_action[tube_id + 1] = self.ext_action_value
        observation, reward, done, info = self.env.step(ext_action)
        self.env.render()

    def rotate_tube(self, tube_id):
        rot_action = np.zeros_like(self.env.action_space.high)
        rot_action[tube_id] = self.rot_action_value
        observation, reward, done, info = self.env.step(np.radians(rot_action))
        self.env.render()

    def full_rotate_tube(self, tube_id):
        for rot in range(self.num_steps[tube_id]):
            self.rotate_tube(tube_id=tube_id)

    def explore_one_tube(self):
        self._init_one_tube()
        observation = self.env.reset()
        # Extend first tube
        for ext in range(self.num_steps[1]):
            self.ext_tube(tube_id=0)
            print('extension count: ', ext)
            # Rotate first tube
            self.full_rotate_tube(tube_id=0)

    def explore_two_tube(self):
        self.__init_two_tube()
        observation = self.env.reset()
        # Extend outer tube
        # Rotate outer tube
        for y in range (self.num_steps[2]):
            print('Rotating outer tube 1 step ... ', y, ' / ', self.num_steps[2])
            self.rotate_tube(tube_id=2)
            if y % 2 > 0:
                print('Retracting and rotating inner tube... ', y, ' / ', self.num_steps[1])
            else:
                print('Extending and rotating inner tube... ', y, ' / ', self.num_steps[1])
            for _ in range(self.num_steps[1]):
                self.ext_tube(tube_id=0, retract=y % 2 > 0)
                self.full_rotate_tube(tube_id=0)
        # Extend out the inner tube so outer tube can be extended
        self.ext_tube(tube_id=0)


import argparse
if __name__ == '__main__':
    # Import the args
    parser = argparse.ArgumentParser()
    parser.add_argument("--numtubes", type=int, help="Number of tubes, either one or two.", default=1)
    parser.add_argument("--extensionvalue", type=float, help="How much to extend at each step.", default=0.005)
    parser.add_argument("--rotationvalue", type=float, help="How much to rotate at each step.", default=30)
    parser.add_argument("--record", type=bool, help="rosbag record", default=False)

    args = parser.parse_args()

    if args.record:
        if args.numtubes == 1:
            rosbag_proc = subprocess.Popen(shlex.split("rosbag record --all -O one_tube_workspace.bag"))
        elif args.numtubes == 2:
            rosbag_proc = subprocess.Popen(shlex.split("rosbag record --all -O two_tube_workspace.bag"))
        else:
            print("Incorect number of tubes chosen.")

    workspace_agent = WorkspaceAgent(max_ext_action=args.extensionvalue, max_rot_action=args.rotationvalue)
    workspace_agent.explore_one_tube()

    if args.numtubes == 1:
        workspace_agent.explore_one_tube()
    elif args.numtubes == 2:
        workspace_agent.explore_two_tube()
    else:
        print("Incorect number of tubes chosen.")

    os.system("rosnode kill --all")
    time.sleep(2.0)

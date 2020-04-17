from pynput.keyboard import Key, Listener
import gym
import ctr_envs
import ctm2_envs

import time

import numpy as np

import argparse


class TeleopAgent(object):
    def __init__(self):
        self.num_tubes = 3
        # self.env = gym.make('Distal-' + str(self.num_tubes) + '-Tube-Reach-v1', ros_flag=True, render_type='human')
        self.env = gym.make('Exact-Ctr-' + str(self.num_tubes) + '-Tube-Reach-v0', ros_flag=True)
        self.key_listener = Listener(on_press=self.on_press_callback)
        self.key_listener.start()

        self.action = np.zeros_like(self.env.action_space.low)
        self.extension_actions = np.zeros(self.num_tubes)
        self.rotation_actions = np.zeros(self.num_tubes)

        self.extension_value = self.env.action_space.high[0] / 2
        self.rotation_value = self.env.action_space.high[-1] / 2
        self.exit = False

    def on_press_callback(self, key):
        # Tube 1 (inner most tube) is w s a d
        # Tube 2 (outer most tube) is t g f h
        # Tube 3 (outer most tube) is i k j l
        try:
            if key.char in ['w', 's', 'a', 'd']:
                if key.char == 'w':
                    self.extension_actions[0] = self.extension_value
                elif key.char == 's':
                    self.extension_actions[0] = -self.extension_value
                elif key.char == 'a':
                    self.rotation_actions[0] = self.rotation_value
                elif key.char == 'd':
                    self.rotation_actions[0] = -self.rotation_value
            if self.num_tubes > 1:
                if key.char in ['t', 'g', 'f', 'h']:
                    if key.char == 't':
                        self.extension_actions[1] = self.extension_value
                    elif key.char == 'g':
                        self.extension_actions[1] = -self.extension_value
                    elif key.char == 'f':
                        self.rotation_actions[1] = self.rotation_value
                    elif key.char == 'h':
                        self.rotation_actions[1] = -self.rotation_value
            if self.num_tubes > 2:
                if key.char in ['i', 'k', 'j', 'l']:
                    if key.char == 'i':
                        self.extension_actions[2] = self.extension_value
                    elif key.char == 'k':
                        self.extension_actions[2] = -self.extension_value
                    elif key.char == 'j':
                        self.rotation_actions[2] = self.rotation_value
                    elif key.char == 'l':
                        self.rotation_actions[2] = -self.rotation_value
        except AttributeError:
            if key == Key.esc:
                self.exit = True
                exit()
            else:
                self.extension_actions = np.zeros(self.num_tubes)
                self.rotation_actions = np.zeros(self.num_tubes)

    def run(self):
        obs = self.env.reset()
        while not self.exit:
            self.action[0:self.num_tubes] = self.extension_actions
            self.action[self.num_tubes:] = self.rotation_actions
            # print('action: ', self.action)
            observation, reward, done, info = self.env.step(self.action)
            self.extension_actions = np.zeros(self.num_tubes)
            self.rotation_actions = np.zeros(self.num_tubes)
            self.env.render()
            if info['is_success']:
                obs = self.env.reset()
            self.action = np.zeros_like(self.env.action_space.low)
            time.sleep(0.1)
        self.env.close()


if __name__ == '__main__':
    teleop_agent = TeleopAgent()
    teleop_agent.run()

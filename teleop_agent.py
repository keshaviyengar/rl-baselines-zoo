from pynput.keyboard import Key, Listener
import gym
import ctm2_envs

import time

import numpy as np


class TeleopAgent:
    def __init__(self):
        self.env = gym.make("Distal-2-Tube-Reach-v0", ros_flag=True, render_type='human')
        self.key_listener = Listener(on_press=self.on_press_callback)
        self.key_listener.start()

        self.action = np.zeros_like(self.env.action_space.low)
        print(self.action)
        self.exit = False

    def on_press_callback(self, key):
        # Tube 1 (inner most tube) is w s a d
        # Tube 2 (outer most tube) is up down left right
        # up is action to extend tube
        # down is action to de-extend tube
        # left is action to rotate tube left
        # right is action to rotate tube right
        try:
            if key.char in ['w', 's', 'a', 'd']:
                if key.char == 'w':
                    self.action[1] = self.env.action_space.high[1] / 2
                elif key.char == 's':
                    self.action[1] = self.env.action_space.low[1] / 2
                elif key.char == 'a':
                    self.action[0] = self.env.action_space.low[0] / 2
                elif key.char == 'd':
                    self.action[0] = self.env.action_space.high[0] / 2
            if key.char in ['i', 'k', 'j', 'l'] and self.env.num_tubes >= 3:
                if key.char == 'i':
                    self.action[5] = self.env.action_space.high[5] / 2
                elif key.char == 'k':
                    self.action[5] = self.env.action_space.low[5] / 2
                elif key.char == 'j':
                    self.action[4] = self.env.action_space.low[4] / 2
                elif key.char == 'l':
                    self.action[4] = self.env.action_space.high[4] / 2
            if key.char in ['t', 'g', 'f', 'h'] and self.env.num_tubes >= 2:
                if key.char == 't':
                    self.action[3] = self.env.action_space.high[3] / 2
                elif key.char == 'g':
                    self.action[3] = self.env.action_space.low[3] / 2
                elif key.char == 'f':
                    self.action[2] = self.env.action_space.low[2] / 2
                elif key.char == 'h':
                    self.action[2] = self.env.action_space.high[2] / 2
        except AttributeError:
            if key == Key.esc:
                self.exit = True
                exit()
            if self.env.num_tubes >= 4:
                if key == Key.up:
                    self.action[7] = self.env.action_space.high[7] / 2
                elif key == Key.down:
                    self.action[7] = self.env.action_space.low[7] / 2
                elif key == Key.left:
                    self.action[6] = self.env.action_space.low[6] / 2
                elif key == Key.right:
                    self.action[6] = self.env.action_space.high[6] / 2
                else:
                    self.action = np.zeros_like(self.env.action_space.low)
            else:
                self.action = np.zeros_like(self.env.action_space.low)

    def run(self):
        max_achieved_goal = np.array([0,0,0])
        observation = self.env.reset()
        while not self.exit:
            self.env.render()
            action = self.action
            observation, reward, done, info = self.env.step(action)
            #max_achieved_goal = np.maximum(max_achieved_goal, observation['achieved_goal'])
            #print('achieved position: ', observation['achieved_goal'])
            #print('max achieved goal: ', max_achieved_goal)
            if info['is_success']:
                observation = self.env.reset()
            # Reset actions to zero
            self.action = np.zeros_like(self.env.action_space.low)
            time.sleep(0.01)
        self.env.close()


if __name__ == '__main__':
    teleop_agent = TeleopAgent()
    teleop_agent.run()

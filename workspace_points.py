import gym
import ctm2_envs
import rospy

import numpy as np
import pandas as pd


class WorkspaceAgent(object):
    def __init__(self, num_tubes):
        self.num_tubes = num_tubes
        if num_tubes == 2:
            self.env = gym.make("Distal-2-Tube-Reach-v0")
        if num_tubes == 3:
            self.env = gym.make("Distal-3-Tube-Reach-v0")
        if num_tubes == 4:
            self.env = gym.make("Distal-4-Tube-Reach-v0")
        self.env.reset()
        self.rotate_action = np.deg2rad(2)
        self.extension_action = 0.0001

    # Rotate a tube given a tube id (inner most is first)
    def rotate_tube(self, tube):
        action = np.zeros_like(self.env.action_space.low)
        if tube == 0:
            action[0] = self.rotate_action
        if tube == 1:
            action[2] = self.rotate_action
        if tube == 2:
            action[4] = self.rotate_action
        if tube == 3:
            action[6] = self.rotate_action
        return action

    def extend_tube(self, tube, retract=False):
        action = np.zeros_like(self.env.action_space.low)
        if tube == 0:
            action[1] = self.extension_action
        if tube == 1:
            action[3] = self.extension_action
        if tube == 2:
            action[5] = self.extension_action
        if tube == 3:
            action[7] = self.extension_action
        if retract:
            action = -1 * action
        return action

    def execute_step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation['achieved_goal']

    def get_two_tube_workspace(self):
        # Tube 1 extension and rotation iterators
        ext_1_itr = int(self.env.tube_length[1] / self.extension_action)
        rot_1_itr = int(np.pi / self.rotate_action)

        # Tube 0 extension and rotation iterators
        ext_0_itr = int(self.env.tube_length[0] / self.extension_action)
        rot_0_itr = int(np.pi / self.rotate_action * 0.5)

        # tube 0 points
        tube_0_points = np.empty([ext_0_itr * ext_0_itr, 3])
        tube_0_count = 0

        # tube 1 points
        tube_1_points = np.empty([ext_0_itr, 3])
        tube_1_count = 0

        # Retract tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_0_count += 1

        # Rotate tube 0
        for rot_0 in range(rot_0_itr):
            self.execute_step(self.rotate_tube(tube=0))

        # Extend tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0))
            tube_0_count += 1

        # Retract tube 0
        for ext_0 in range(ext_0_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))

        # Retract tube 1
        for ext_1 in range(ext_1_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_1_points[tube_1_count, :] = self.execute_step(self.extend_tube(tube=1, retract=True))
            self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_1_count += 1

        return tube_0_points, tube_1_points

    def get_three_tube_workspace(self):
        # Tube 2 extension and rotation iterators
        ext_2_itr = int(self.env.tube_length[2] / self.extension_action)
        rot_2_itr = int(np.pi / self.rotate_action * 0.5)

        # Tube 1 extension and rotation iterators
        ext_1_itr = int(self.env.tube_length[1] / self.extension_action)
        rot_1_itr = int(np.pi / self.rotate_action * 0.5)

        # Tube 0 extension and rotation iterators
        ext_0_itr = int(self.env.tube_length[0] / self.extension_action)
        rot_0_itr = int(np.pi / self.rotate_action * 0.5)

        # tube 0 points
        tube_0_points = np.empty([4 * ext_0_itr, 3])
        tube_0_count = 0

        # tube 1 points
        tube_1_points = np.empty([2 * ext_1_itr, 3])
        tube_1_count = 0

        # tube 2 points
        tube_2_points = np.empty([ext_2_itr, 3])
        tube_2_count = 0

        # Retract tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_0_count += 1

        # Rotate tube 0
        for rot_0 in range(rot_0_itr):
            self.execute_step(self.rotate_tube(tube=0))

        # Extend tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0))
            tube_0_count += 1

        # Retract tube 0
        for ext_0 in range(ext_0_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))

        # Rotate tube 1
        for rot_1 in range(rot_1_itr):
            self.execute_step(self.rotate_tube(tube=1))

        # Extend tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0))
            tube_0_count += 1

        # Retract tube 0
        for ext_0 in range(ext_0_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))

        # Rotate tube 0
        for rot_0 in range(rot_0_itr):
            self.execute_step(self.rotate_tube(tube=0))

        # Extend tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0))
            tube_0_count += 1

        # Retract tube 0
        for ext_0 in range(ext_0_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))

        # Retract tube 1
        for ext_1 in range(ext_1_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_1_points[tube_1_count, :] = self.execute_step(self.extend_tube(tube=1, retract=True))
            self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_1_count += 1

        # Rotate tube 1
        for rot_1 in range(rot_1_itr):
            self.execute_step(self.rotate_tube(tube=1))

        # Extend tube 1
        for ext_1 in range(ext_1_itr):
            tube_1_points[tube_1_count, :] = self.execute_step(self.extend_tube(tube=1))
            tube_1_count += 1

        # Retract tube 1
        for ext_1 in range(ext_1_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))
            self.execute_step(self.extend_tube(tube=1, retract=True))
            self.execute_step(self.extend_tube(tube=0, retract=True))

        # Retract tube 2
        for ext_2 in range(ext_2_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))
            self.execute_step(self.extend_tube(tube=1, retract=True))
            tube_2_points[tube_2_count, :] = self.execute_step(self.extend_tube(tube=2, retract=True))
            self.execute_step(self.extend_tube(tube=0, retract=True))
            self.execute_step(self.extend_tube(tube=1, retract=True))
            tube_2_count += 1

        return tube_0_points, tube_1_points, tube_2_points

    def get_four_tube_workspace(self):
        # Tube 3 extension and rotation iterators
        ext_3_itr = int(self.env.tube_length[3] / self.extension_action)
        rot_3_itr = int(np.pi / self.rotate_action * 0.5)

        # Tube 2 extension and rotation iterators
        ext_2_itr = int(self.env.tube_length[2] / self.extension_action)
        rot_2_itr = int(np.pi / self.rotate_action * 0.5)

        # Tube 1 extension and rotation iterators
        ext_1_itr = int(self.env.tube_length[1] / self.extension_action)
        rot_1_itr = int(np.pi / self.rotate_action * 0.5)

        # Tube 0 extension and rotation iterators
        ext_0_itr = int(self.env.tube_length[0] / self.extension_action)
        rot_0_itr = int(np.pi / self.rotate_action * 0.5)

        # tube 0 points
        tube_0_points = np.empty([8 * ext_0_itr, 3])
        tube_0_count = 0

        # tube 1 points
        tube_1_points = np.empty([4 * ext_1_itr, 3])
        tube_1_count = 0

        # tube 2 points
        tube_2_points = np.empty([3 * ext_2_itr, 3])
        tube_2_count = 0

        # tube 3 points
        tube_3_points = np.empty([ext_3_itr, 3])
        tube_3_count = 0

        # Retract tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_0_count += 1

        # Retract tube 1
        for ext_1 in range(ext_1_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_1_points[tube_1_count, :] = self.execute_step(self.extend_tube(tube=1, retract=True))
            self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_1_count += 1

        # Extend tube 1
        for ext_1 in range(ext_1_itr):
            self.execute_step(self.extend_tube(tube=1))

        # Rotate tube 0
        for rot_0 in range(rot_0_itr):
            self.execute_step(self.rotate_tube(tube=0))

        # Extend tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0))
            tube_0_count += 1

        # Retract tube 0
        for ext_0 in range(ext_0_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))

        # Rotate tube 1
        for rot_1 in range(rot_1_itr):
            self.execute_step(self.rotate_tube(tube=1))

        # Extend tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0))
            tube_0_count += 1

        # Rotate tube 0
        for rot_0 in range(rot_0_itr):
            self.execute_step(self.rotate_tube(tube=0))

        # Retract tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_0_count += 1

        # Retract tube 1
        for ext_1 in range(ext_1_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_1_points[tube_1_count, :] = self.execute_step(self.extend_tube(tube=1, retract=True))
            self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_1_count += 1

        # Retract tube 2
        for ext_2 in range(ext_2_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))
            self.execute_step(self.extend_tube(tube=1, retract=True))
            tube_2_points[tube_2_count, :] = self.execute_step(self.extend_tube(tube=2, retract=True))
            self.execute_step(self.extend_tube(tube=0, retract=True))
            self.execute_step(self.extend_tube(tube=1, retract=True))
            tube_2_count += 1

        # Rotate tube 2
        for rot_2 in range(rot_2_itr):
            self.execute_step(self.rotate_tube(tube=2))

        # Extend tube 2
        for ext_2 in range(ext_2_itr):
            tube_2_points[tube_2_count, :] = self.execute_step(self.extend_tube(tube=2))
            tube_2_count += 1

        # Extend tube 1
        for ext_1 in range(ext_1_itr):
            tube_1_points[tube_1_count, :] = self.execute_step(self.extend_tube(tube=1))
            tube_1_count += 1

        # Extend tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0))
            tube_0_count += 1

        # Rotate tube 0
        for rot_0 in range(rot_0_itr):
            self.execute_step(self.rotate_tube(tube=0))

        # Retract tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_0_count += 1

        # Rotate tube 1
        for rot_1 in range(rot_1_itr):
            self.execute_step(self.rotate_tube(tube=1))

        # Extend tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0))
            tube_0_count += 1

        # Rotate tube 0
        for rot_0 in range(rot_0_itr):
            self.execute_step(self.rotate_tube(tube=0))

        # Retract tube 0
        for ext_0 in range(ext_0_itr):
            tube_0_points[tube_0_count, :] = self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_0_count += 1

        # Retract tube 1
        for ext_1 in range(ext_1_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_1_points[tube_1_count, :] = self.execute_step(self.extend_tube(tube=1, retract=True))
            self.execute_step(self.extend_tube(tube=0, retract=True))
            tube_1_count += 1

        # Retract tube 2
        for ext_2 in range(ext_2_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))
            self.execute_step(self.extend_tube(tube=1, retract=True))
            tube_2_points[tube_2_count, :] = self.execute_step(self.extend_tube(tube=2, retract=True))
            self.execute_step(self.extend_tube(tube=0, retract=True))
            self.execute_step(self.extend_tube(tube=1, retract=True))
            tube_2_count += 1

        # Retract tube 3
        for ext_3 in range(ext_3_itr):
            self.execute_step(self.extend_tube(tube=0, retract=True))
            self.execute_step(self.extend_tube(tube=1, retract=True))
            self.execute_step(self.extend_tube(tube=2, retract=True))
            tube_3_points[tube_3_count, :] = self.execute_step(self.extend_tube(tube=3, retract=True))
            self.execute_step(self.extend_tube(tube=0, retract=True))
            self.execute_step(self.extend_tube(tube=1, retract=True))
            self.execute_step(self.extend_tube(tube=2, retract=True))
            tube_3_count += 1

        return tube_0_points, tube_1_points, tube_2_points, tube_3_points


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    num_tubes = 4
    agent = WorkspaceAgent(num_tubes=num_tubes)
    if num_tubes == 2:
        tube_0_points, tube_1_points = agent.get_two_tube_workspace()

        tube_0_df = pd.DataFrame(tube_0_points * 1000, columns=['x', 'y', 'z'])
        tube_1_df = pd.DataFrame(tube_1_points * 1000, columns=['x', 'y', 'z'])

        ax = tube_0_df.plot(kind='scatter', x='x', y='z', color='r', label='tube 0')
        tube_1_df.plot(kind='scatter', x='x', y='z', color='g', ax=ax, label='tube 1')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('z (mm)')

        plt.axis('equal')
        plt.savefig('2-tube-workspace.png', dpi=300)

    if num_tubes == 3:
        tube_0_points, tube_1_points, tube_2_points = agent.get_three_tube_workspace()

        tube_0_df = pd.DataFrame(tube_0_points * 1000, columns=['x', 'y', 'z'])
        tube_1_df = pd.DataFrame(tube_1_points * 1000, columns=['x', 'y', 'z'])
        tube_2_df = pd.DataFrame(tube_2_points * 1000, columns=['x', 'y', 'z'])

        ax = tube_0_df.plot(kind='scatter', x='x', y='z', color='r', label='tube 0')
        tube_1_df.plot(kind='scatter', x='x', y='z', color='g', ax=ax, label='tube 1')
        tube_2_df.plot(kind='scatter', x='x', y='z', color='b', ax=ax, label='tube 2')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('z (mm)')

        plt.axis('equal')
        plt.savefig('3-tube-workspace.png', dpi=300)

    if num_tubes == 4:
        tube_0_points, tube_1_points, tube_2_points, tube_3_points = agent.get_four_tube_workspace()

        tube_0_df = pd.DataFrame(tube_0_points * 1000, columns=['x', 'y', 'z'])
        tube_1_df = pd.DataFrame(tube_1_points * 1000, columns=['x', 'y', 'z'])
        tube_2_df = pd.DataFrame(tube_2_points * 1000, columns=['x', 'y', 'z'])
        tube_3_df = pd.DataFrame(tube_3_points * 1000, columns=['x', 'y', 'z'])

        ax = tube_0_df.plot(kind='scatter', x='x', y='z', color='r', label='tube 0')
        tube_1_df.plot(kind='scatter', x='x', y='z', color='g', ax=ax, label='tube 1')
        tube_2_df.plot(kind='scatter', x='x', y='z', color='b', ax=ax, label='tube 2')
        tube_3_df.plot(kind='scatter', x='x', y='z', color='k', ax=ax, label='tube 3')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('z (mm)')

        plt.axis('equal')
        plt.savefig('4-tube-workspace.png', dpi=300)

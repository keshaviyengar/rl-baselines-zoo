import gym
import ctm2_envs

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# This script samples uniformly from the joint space of an environment, performs FK, and returns the Cartesian point.
# Saves as a pandas dataframe

plt.interactive(True)

if __name__ == '__main__':
    env = gym.make("Distal-4-Tube-Reach-v0", ros_flag=False)

    num_tubes = 4
    num_samples = 1000
    alpha_joint_data = np.empty([num_samples, 2 * num_tubes])
    beta_joint_data = np.empty([num_samples, num_tubes])
    ext_joint_data = np.empty([num_samples, num_tubes])
    rot_joint_data = np.empty([num_samples, num_tubes])
    cartesian_data = np.empty([num_samples, 3])
    for itr in range(num_samples):
        cartesian_sample, triplet_joint_sample, joint_sample = env.sample_goal()
        cartesian_data[itr, :] = cartesian_sample

        if num_tubes == 2:
            beta_joint_data[itr, :] = [triplet_joint_sample[2], triplet_joint_sample[5]]
            alpha_joint_data[itr, :] = [triplet_joint_sample[0], triplet_joint_sample[1], triplet_joint_sample[3],
                                        triplet_joint_sample[4]]
            rot_joint_data[itr, :] = [joint_sample[0], joint_sample[2]]
            ext_joint_data[itr, :] = [joint_sample[1], joint_sample[3]]
        if num_tubes == 3:
            beta_joint_data[itr, :] = [triplet_joint_sample[2], triplet_joint_sample[5], triplet_joint_sample[8]]
            alpha_joint_data[itr, :] = [triplet_joint_sample[0], triplet_joint_sample[1], triplet_joint_sample[3],
                                        triplet_joint_sample[4], triplet_joint_sample[6], triplet_joint_sample[7]]
            rot_joint_data[itr, :] = [joint_sample[0], joint_sample[2], joint_sample[4]]
            ext_joint_data[itr, :] = [joint_sample[1], joint_sample[3], joint_sample[5]]
        if num_tubes == 4:
            beta_joint_data[itr, :] = [triplet_joint_sample[2], triplet_joint_sample[5], triplet_joint_sample[8], triplet_joint_sample[11]]
            alpha_joint_data[itr, :] = [triplet_joint_sample[0], triplet_joint_sample[1], triplet_joint_sample[3],
                                        triplet_joint_sample[4], triplet_joint_sample[6], triplet_joint_sample[7],
                                        triplet_joint_sample[9], triplet_joint_sample[10]]
            rot_joint_data[itr, :] = [joint_sample[0], joint_sample[2], joint_sample[4], joint_sample[6]]
            ext_joint_data[itr, :] = [joint_sample[1], joint_sample[3], joint_sample[5], joint_sample[7]]

    # Create into dataframe for easy visualization
    if num_tubes == 2:
        beta_joint_df = pd.DataFrame(beta_joint_data, columns=['beta_0', 'beta_1'])
        a1 = beta_joint_df.plot.hist(bins=12, alpha=0.5)

        alpha_joint_df = pd.DataFrame(alpha_joint_data, columns=['tube_0_0', 'tube_0_1', 'tube_1_0', 'tube_1_1'])
        a2 = alpha_joint_df.plot.hist(bins=12, alpha=0.5)

        ext_joint_df = pd.DataFrame(ext_joint_data, columns=['ext_0', 'ext_1'])
        a3 = ext_joint_df.plot.hist(bins=12, alpha=0.5)

        rot_joint_df = pd.DataFrame(rot_joint_data, columns=['rot_0', 'rot_1'])
        a4 = rot_joint_df.plot.hist(bins=12, alpha=0.5)

        cart_df = pd.DataFrame(cartesian_data, columns=['x', 'y', 'z'])
        a5 = cart_df.plot.hist(bins=12, alpha=0.5)

    if num_tubes == 3:
        beta_joint_df = pd.DataFrame(beta_joint_data, columns=['beta_0', 'beta_1', 'beta_3'])
        a1 = beta_joint_df.plot.hist(bins=12, alpha=0.5)

        alpha_joint_df = pd.DataFrame(alpha_joint_data, columns=['tube_0_0', 'tube_0_1', 'tube_1_0', 'tube_1_1', 'tube_2_0', 'tube_2_1'])
        a2 = alpha_joint_df.plot.hist(bins=12, alpha=0.5)

        ext_joint_df = pd.DataFrame(ext_joint_data, columns=['ext_0', 'ext_1', 'ext_2'])
        a3 = ext_joint_df.plot.hist(bins=12, alpha=0.5)

        rot_joint_df = pd.DataFrame(rot_joint_data, columns=['rot_0', 'rot_1', 'rot_2'])
        a4 = rot_joint_df.plot.hist(bins=12, alpha=0.5)

        cart_df = pd.DataFrame(cartesian_data, columns=['x', 'y', 'z'])
        a5 = cart_df.plot.hist(bins=12, alpha=0.5)

    if num_tubes == 4:
        beta_joint_df = pd.DataFrame(beta_joint_data, columns=['beta_0', 'beta_1', 'beta_3', 'beta_4'])
        a1 = beta_joint_df.plot.hist(bins=12, alpha=0.5)

        alpha_joint_df = pd.DataFrame(alpha_joint_data, columns=['tube_0_0', 'tube_0_1', 'tube_1_0', 'tube_1_1', 'tube_2_0', 'tube_2_1', 'tube_3_0', 'tube_3_1'])
        a2 = alpha_joint_df.plot.hist(bins=12, alpha=0.5)

        ext_joint_df = pd.DataFrame(ext_joint_data, columns=['ext_0', 'ext_1', 'ext_2', 'ext_3'])
        a3 = ext_joint_df.plot.hist(bins=12, alpha=0.5)

        rot_joint_df = pd.DataFrame(rot_joint_data, columns=['rot_0', 'rot_1', 'rot_2', 'rot_3'])
        a4 = rot_joint_df.plot.hist(bins=12, alpha=0.5)

        cart_df = pd.DataFrame(cartesian_data, columns=['x', 'y', 'z'])
        a5 = cart_df.plot.hist(bins=12, alpha=0.5)

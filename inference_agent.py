import gym
import ctm2_envs

import time
import os

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper
from utils import create_test_env, get_saved_hyperparams

# Import ROS tools
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Point
import rosbag

# ROS node for inferencing
# Subscribes to a point (or other type of topic) to get desired goal
# Uses stable-baselines model to attempt to reach goal from current position


class Ctm2Inference(object):
    def __init__(self):
        env_id = "Distal-2-Tube-Reach-v0"
        seed = np.random.randint(0, 10)

        log_dir = None
        folder = "trained_agents"
        algo = "her"
        log_path = os.path.join(folder, algo)
        model_path = "{}/{}.pkl".format(log_path, env_id)
        stats_path = os.path.join(log_path, env_id)

        assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)
        assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, env_id, model_path)
        set_global_seeds(seed)

        '''
        hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)
        env = create_test_env(env_id, n_envs=1, is_atari=False, stats_path=stats_path, seed=seed, log_dir=log_dir,
                               should_render=True, hyperparams=hyperparams)
        '''
        self.env = HERGoalEnvWrapper(gym.make(env_id, ros_flag=True, render_type='human'))

        self.model = HER.load(model_path, env=self.env)

    def infer_to_goal(self):
        # Initialize stats variables
        episode_reward = 0.0
        episode_rewards = []
        ep_len = 0
        episode_lengths = []
        errors = []
        successes = []
        q_goals = []
        # Create a pandas dataframe for storing data
        df = pd.DataFrame(
            columns=['total steps', 'rewards', 'errors', 'success', 'q goal_1', 'q goal_2', 'q_goal_3', 'q_goal_4'])

        # Set a desired goal here after the reset
        desired_goal = np.array([0, 0, 0.04])

        # Set the observation
        obs = self.env.reset(goal=desired_goal)

        for t in range(200):
            action, _ = self.model.predict(obs, deterministic=True)

            # Ensure action space is of type Box
            if isinstance(self.env.action_space, gym.spaces.Box):
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            obs, reward, done, infos = self.env.step(action)
            self.env.render()
            input("time step done, press enter to continue...")

            episode_reward += reward
            ep_len += 1

            if done or infos.get('is_success', False):
                successes.append(infos.get('is_success', False))
                errors.append(infos.get('error', np.inf))
                episode_lengths.append(ep_len)
                episode_rewards.append(episode_reward)
                q_goals.append(infos.get('q_goal', False))
                break

        df['total steps'] = episode_lengths
        df['rewards'] = episode_rewards
        df['errors'] = errors
        df['success'] = successes
        df['q goal_1'] = [q[0] for q in q_goals]
        df['q goal_2'] = [q[1] for q in q_goals]
        df['q goal_3'] = [q[2] for q in q_goals]
        df['q goal_4'] = [q[3] for q in q_goals]

        print("Final error: ", errors[-1])


if __name__ == '__main__':
    inferencer = Ctm2Inference()
    inferencer.infer_to_goal()


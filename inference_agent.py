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
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
import rosbag

import signal

# ROS node for inferencing
# Subscribes to a point (or other type of topic) to get desired goal
# Uses stable-baselines model to attempt to reach goal from current position


class Ctm2Inference(object):
    def __init__(self, experiment_id, episode_timesteps, goal_tolerance):
        env_id = None
        self.exp_id = experiment_id
        if experiment_id in [1,2,3,4,5]:
            env_id = "Distal-2-Tube-Reach-v0"
        if experiment_id in [6,7,8,9,10]:
            env_id = "Distal-3-Tube-Reach-v0"
        if experiment_id in [11,12,13,14,15]:
            env_id = "Distal-4-Tube-Reach-v0"

        self.save = False

        self.episode_timesteps = episode_timesteps

        self.env = HERGoalEnvWrapper(gym.make(env_id, goal_tolerance=goal_tolerance, ros_flag=True, render_type='human'))
        seed = np.random.randint(0, 10)

        model_path = "/home/keshav/ctm2-stable-baselines/saved_results/" + "exp_" + str(experiment_id) + "/" + env_id + ".pkl"
        self.model = HER.load(model_path, env=self.env)

        set_global_seeds(seed)

        self.desired_goal_sub = rospy.Subscriber("desired_goal", Pose, self.desired_goal_callback)
        self.desired_goal = np.array([0, 0, 0])

        self.trajectory_finish_sub = rospy.Subscriber("trajectory_finish", Bool, self.trajectory_finish_callback)

        # Create a pandas dataframe for storing data
        self.df = pd.DataFrame(
            columns=['achieved_goal_x', 'achieved_goal_y', 'achieved_goal_z',
                     'desired_goal_x', 'desired_goal_y', 'desired_goal_z', 'errors', 'time_taken'])

        # Initialize stats variables for saving
        self.achieved_goals_x = []
        self.achieved_goals_y = []
        self.achieved_goals_z = []
        self.desired_goals_x = []
        self.desired_goals_y = []
        self.desired_goals_z = []
        self.errors = []
        self.times_taken = []

        # Initialize current variables for stats
        self.achieved_goal_x = 0
        self.achieved_goal_y = 0
        self.achieved_goal_z = 0
        self.desired_goal_x = 0
        self.desired_goal_y = 0
        self.desired_goal_z = 0
        self.error = 0
        self.time_taken = 0
        self.start_time = rospy.Time.now()

    def desired_goal_callback(self, msg):
        # Append the stats variables at the callback
        self.achieved_goals_x = np.append(self.achieved_goals_x, self.achieved_goal_x)
        self.achieved_goals_y = np.append(self.achieved_goals_y, self.achieved_goal_y)
        self.achieved_goals_z = np.append(self.achieved_goals_z, self.achieved_goal_z)
        self.desired_goals_x = np.append(self.desired_goals_x, self.desired_goal_x)
        self.desired_goals_y = np.append(self.desired_goals_y, self.desired_goal_y)
        self.desired_goals_z = np.append(self.desired_goals_z, self.desired_goal_z)
        self.errors = np.append(self.errors, self.error)
        self.times_taken = np.append(self.times_taken, self.time_taken)
        self.desired_goal = np.array([msg.position.x / 1000, msg.position.y / 1000, msg.position.z / 1000])

    def trajectory_finish_callback(self, msg):
        if msg.data and not self.save:
            self.save = True
            # End of trajectory, save data frame
            self.df['achieved_goal_x'] = self.achieved_goals_x
            self.df['achieved_goal_y'] = self.achieved_goals_y
            self.df['achieved_goal_z'] = self.achieved_goals_z
            self.df['desired_goal_x'] = self.desired_goals_x
            self.df['desired_goal_y'] = self.desired_goals_y
            self.df['desired_goal_z'] = self.desired_goals_z
            self.df['errors'] = self.errors
            self.df["time_taken"] = self.time_taken
            # self.df.to_csv('~/ctm2-stable-baselines/saved-runs/results/' + 'exp_' + str(self.exp_id) + '/square_traj.csv')
            self.df.to_csv('~/ctm2-stable-baselines/saved_results/exp-14-0-3-square.csv')

    def infer_to_goal(self):
        episode_reward = 0.0
        ep_len = 0

        # Set the observation
        obs = self.env.reset(goal=self.desired_goal)

        for t in range(self.episode_timesteps):
            action, _ = self.model.predict(obs, deterministic=True)

            # Ensure action space is of type Box
            if isinstance(self.env.action_space, gym.spaces.Box):
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            obs, reward, done, infos = self.env.step(action)
            self.env.render()

            episode_reward += reward
            ep_len += 1

            self.achieved_goal_x = self.env.convert_obs_to_dict(obs)['achieved_goal'][0]
            self.achieved_goal_y = self.env.convert_obs_to_dict(obs)['achieved_goal'][1]
            self.achieved_goal_z = self.env.convert_obs_to_dict(obs)['achieved_goal'][2]
            self.desired_goal_x = self.env.convert_obs_to_dict(obs)['desired_goal'][0]
            self.desired_goal_y = self.env.convert_obs_to_dict(obs)['desired_goal'][1]
            self.desired_goal_z = self.env.convert_obs_to_dict(obs)['desired_goal'][2]
            self.error = infos['error']
            self.time_taken = (rospy.Time.now() - self.start_time).to_sec()
            if done or infos.get('is_success', False):
                break


if __name__ == '__main__':
    experiment_id = 4
    episode_timesteps = 150
    inferencer = Ctm2Inference(experiment_id=experiment_id, episode_timesteps=episode_timesteps, goal_tolerance=0.0003)
    while True:
        inferencer.infer_to_goal()
        if rospy.is_shutdown():
            break


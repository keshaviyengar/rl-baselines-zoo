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
import rosbag

import signal

# ROS node for inferencing
# Subscribes to a point (or other type of topic) to get desired goal
# Uses stable-baselines model to attempt to reach goal from current position


class Ctm2Inference(object):
    def __init__(self):
        env_id = "Distal-2-Tube-Reach-v0"
        #env_id = "Distal-1-Tube-Reach-v0"
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

        signal.signal(signal.SIGINT, handler=self._ctrl_c_handler)

        '''
        hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)
        env = create_test_env(env_id, n_envs=1, is_atari=False, stats_path=stats_path, seed=seed, log_dir=log_dir,
                               should_render=True, hyperparams=hyperparams)
        '''
        self.env = HERGoalEnvWrapper(gym.make(env_id, ros_flag=True, render_type='human'))
        self.model = HER.load(model_path, env=self.env)

        self.desired_goal_sub = rospy.Subscriber("desired_goal", Pose, self.desired_goal_callback)
        self.desired_goal = np.array([0, 0, 0])

        # Create a pandas dataframe for storing data
        self.df = pd.DataFrame(
            columns=['total steps', 'rewards', 'errors', 'success', 'time_taken', 'q goal_1', 'q goal_2', 'q_goal_3', 'q_goal_4'])

        # Initialize stats variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.errors = []
        self.successes = []
        self.q_goals = []
        self.time_taken = []

    def desired_goal_callback(self, msg):
        self.desired_goal = np.array([msg.position.x / 1000, msg.position.y / 1000, msg.position.z / 1000])

    def publish_pcl(self):
        pass

    def infer_to_goal(self):
        episode_reward = 0.0
        ep_len = 0

        # Set the observation
        obs = self.env.reset(goal=self.desired_goal)

        start_time = rospy.Time.now()
        for t in range(200):
            action, _ = self.model.predict(obs, deterministic=True)

            # Ensure action space is of type Box
            if isinstance(self.env.action_space, gym.spaces.Box):
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            obs, reward, done, infos = self.env.step(action)
            self.env.render()

            episode_reward += reward
            ep_len += 1

            if done or infos.get('is_success', False):
                self.successes.append(infos.get('is_success', False))
                self.errors.append(infos.get('error', np.inf))
                self.episode_lengths.append(ep_len)
                self.episode_rewards.append(episode_reward)
                self.q_goals.append(infos.get('q_goal', False))
                self.time_taken.append((rospy.Time.now() - start_time).to_sec())
                break

    def _ctrl_c_handler(self,  signal, frame):
        print("Ctrl c pressed!")
        self.df['total steps'] = self.episode_lengths
        self.df['rewards'] = self.episode_rewards
        self.df['errors'] = self.errors
        self.df['success'] = self.successes
        self.df["time_taken"] = self.time_taken
        self.df.to_csv("2-tube.csv")


if __name__ == '__main__':
    inferencer = Ctm2Inference()
    while True:
        inferencer.infer_to_goal()
        if rospy.is_shutdown():
            break


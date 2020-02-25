import gym
import ctm2_envs

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines.her.utils import HERGoalEnvWrapper
from stable_baselines import DDPG, HER


class Ctm2Evaluation(object):
    def __init__(self, experiment_id, goal_tolerance):
        env_id = None
        if experiment_id in [1,2,3,4,5]:
            env_id = "Distal-2-Tube-Reach-v0"
        if experiment_id in [6,7,8,9, 10]:
            env_id = "Distal-3-Tube-Reach-v0"
        if experiment_id in [11,12,13,14,15]:
            env_id = "Distal-4-Tube-Reach-v0"

        self.env = HERGoalEnvWrapper(gym.make(env_id, goal_tolerance=goal_tolerance))

        model_path = "/home/keshav/ctm2-stable-baselines/saved_results/" + "exp_" + str(experiment_id) + "/" + env_id + ".pkl"
        self.model = HER.load(model_path, env=self.env)

    def evaluate(self, num_timesteps):
        joint_1_extension_errors = []
        joint_2_extension_errors = []
        joint_3_extension_errors = []
        episode_lengths = []
        episode_rewards = []

        episode_reward = 0.0
        ep_len = 0
        obs = self.env.reset()

        for t in range(num_timesteps):
            action, _ = self.model.predict(obs, deterministic=True)

            # Ensure action space is of type Box
            if isinstance(self.env.action_space, gym.spaces.Box):
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            obs, reward, done, infos = self.env.step(action)
            self.env.render()

            episode_reward += reward
            ep_len += 1

            if done or infos.get('is_success', False):
                if infos.get('is_success'):
                    rotation_1_error = np.linalg.norm(infos.get('q_achieved')[0] - infos.get('q_goal')[0])
                    if rotation_1_error < 0.01:
                        joint_1_extension_errors.append(rotation_1_error)
                    rotation_2_error = np.linalg.norm(infos.get('q_achieved')[2] - infos.get('q_goal')[2])
                    if rotation_2_error < 0.01:
                        joint_2_extension_errors.append(rotation_2_error)
                    rotation_3_error = np.linalg.norm(infos.get('q_achieved')[4] - infos.get('q_goal')[4])
                    if rotation_3_error < 0.01:
                        joint_3_extension_errors.append(rotation_3_error)
                # Reset to new goal
                ep_len = 0
                obs = self.env.reset()

        print(np.mean(joint_1_extension_errors))
        print(np.mean(joint_2_extension_errors))
        print(np.mean(joint_3_extension_errors))
        return successes, errors, episode_lengths, episode_rewards


if __name__ == '__main__':
    goal_tolerances = [0.0004]
    goal_tolerance_array = []
    mean_ep_length_array = []
    success_array = []
    mean_error_array = []
    std_error_array = []

    exp_df = pd.DataFrame(columns=['goal_tolerance', 'mean_ep_length', 'success_rate', 'mean_error', 'std_error'])
    for goal_tolerance in goal_tolerances:
        evaluator = Ctm2Evaluation(experiment_id=10, goal_tolerance=goal_tolerance)
        successes, errors, episode_lengths, episode_rewards = evaluator.evaluate(num_timesteps=5000)
        mean_ep_length_array.append(np.mean(episode_lengths))
        goal_tolerance_array.append(goal_tolerance)
        success_array.append(np.mean(successes))
        mean_error_array.append(np.mean(errors))
        std_error_array.append(np.mean(errors))

        print("current finished tolerance: ", str(goal_tolerance))

    exp_df['goal_tolerance'] = goal_tolerance_array
    exp_df['mean_ep_length'] = mean_ep_length_array
    exp_df['success_rate'] = success_array
    exp_df['mean_error'] = mean_error_array
    exp_df['std_error'] = std_error_array

    exp_df.to_csv('~/ctm2-stable-baselines/saved-runs/results/exp_9/eval.csv')

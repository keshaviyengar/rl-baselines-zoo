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


if __name__ == '__main__':
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

    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

    env = create_test_env(env_id, n_envs=1, is_atari=False, stats_path=stats_path, seed=seed, log_dir=log_dir,
                          should_render=False, hyperparams=hyperparams)

    model = HER.load(model_path, env=env)

    obs = env.reset()
    episode_reward = 0.0
    episode_rewards = []
    ep_len = 0
    episode_lengths = []
    errors = []
    successes = []
    q_goals = []

    #df = pd.DataFrame(columns=['total steps', 'rewards', 'errors', 'success', 'q goal_1', 'q goal_2'])
    df = pd.DataFrame(columns=['total steps', 'rewards', 'errors', 'success', 'q goal_1', 'q goal_2', 'q_goal_3', 'q_goal_4'])

    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)

        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)

        episode_reward += reward[0]
        ep_len += 1

        if done or infos[0].get('is_success', False):
            obs = env.reset()
            successes.append(infos[0].get('is_success', False))
            errors.append(infos[0].get('error', np.inf))
            episode_lengths.append(ep_len)
            episode_rewards.append(episode_reward)
            q_goals.append(infos[0].get('q_goal', False))
            episode_reward, ep_len = 0.0, 0

    df['total steps'] = episode_lengths
    df['rewards'] = episode_rewards
    df['errors'] = errors
    df['success'] = successes
    df['q goal_1'] = [q[0] for q in q_goals]
    df['q goal_2'] = [q[1] for q in q_goals]
    df['q goal_3'] = [q[2] for q in q_goals]
    df['q goal_4'] = [q[3] for q in q_goals]

    df.to_csv('test2.csv')

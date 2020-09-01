import os
import numpy as np
from collections import OrderedDict

from pypcd import pypcd
import h5py

# Same code as ctm_callback but adapted for different states from PPO2 algorithm


class CtmPPOCallback(object):
    def __init__(self, log_folder, n_envs, n_timesteps, inc_goals_obs, obs_dim, goal_dim, goal_tolerance_parameters=None):
        self.log_folder = log_folder
        self.n_timesteps = n_timesteps
        self.n_envs = n_envs

        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.inc_goals_obs = inc_goals_obs
        if self.inc_goals_obs:
            # Important: gym mixes up ordered and unordered keys
            # and the Dict space may return a different order of keys that the actual one
            self.KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']
        else:
            self.KEY_ORDER = ['observation']

        self.goal_tolerance_parameters = goal_tolerance_parameters

        # Keep storage as list, then convert to numpy when saving (more efficent).
        self.ag_points = []
        self.dg_points = []
        self.errors = []

        self.callback_save_count = 0
        self.save_intervals = 10

        os.makedirs(log_folder + '/learned_policy', exist_ok=True)
        os.makedirs(log_folder + '/saved_data', exist_ok=True)
        os.makedirs(log_folder + '/point_clouds', exist_ok=True)

    def callback(self, _locals, _globals):
        self.callback_save_count += 1

        ep_infos = _locals['ep_infos']
        ag_points = [ep['achieved_goal'] for ep in ep_infos]
        dg_points = [ep['desired_goal'] for ep in ep_infos]
        errors = [ep['error'] for ep in ep_infos]

        self.ag_points.extend(ag_points * 1000)
        self.dg_points.extend(dg_points * 1000)
        self.errors.extend(errors)

        # update goal tolerance if needed. TODO: env functions not accessible with DummyVec Wrapper
        # _locals['self'].env.env.update_goal_tolerance(self.training_step)

        if (self.callback_save_count % self.save_intervals) == 0:
            self.convert_to_np_arrays()
            self.save_np_arrays(_locals['timestep'])
            self.clear_np_arrays()
            # Save model at intervals
            _locals['self'].save(self.log_folder + '/learned_policy/' + str(_locals['timestep']) + '_saved_model.pkl')

    def convert_dict_to_obs(self, obs_dict):
        return np.concatenate([obs_dict[key] for key in self.KEY_ORDER])

    def convert_obs_to_dict(self, observations):
        if self.goal_dim == 0:
            return OrderedDict([
                ('observation', observations[:self.obs_dim]),
            ])
        else:
            return OrderedDict([
                ('observation', observations[:self.obs_dim]),
                ('achieved_goal', observations[self.obs_dim:self.obs_dim + self.goal_dim]),
                ('desired_goal', observations[self.obs_dim + self.goal_dim:]),
            ])

    def convert_to_np_arrays(self):
        self.ag_points = np.asarray(self.ag_points)
        self.dg_points = np.asarray(self.dg_points)
        self.errors = np.asarray(self.errors)

    def save_np_arrays(self, timestep):
        hf = h5py.File(self.log_folder + '/saved_data/' + 'data_' + str(timestep) + '.h5', 'w')
        hf.create_dataset('achieved_goals', data=self.ag_points)
        hf.create_dataset('desired_goals', data=self.dg_points)
        hf.create_dataset('errors', data=self.errors)
        hf.close()

    def clear_np_arrays(self):
        self.ag_points = []
        self.dg_points = []
        self.errors = []

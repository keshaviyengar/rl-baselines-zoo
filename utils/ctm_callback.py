import os
import numpy as np
from collections import OrderedDict
from mpi4py import MPI

from pypcd import pypcd
import h5py


class CtmCallback(object):
    def __init__(self, log_folder, n_timesteps, inc_goals_obs, obs_dim, goal_dim, goal_tolerance_parameters=None):
        self.log_folder = log_folder
        self.n_timesteps = n_timesteps
        self.num_workers = 1
        self.rank = None
        self.num_workers = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()

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

        self.ag_points = []
        self.dg_points = []
        self.errors = []

        self.training_step = 0
        self.save_intervals = np.arange(0, n_timesteps + 1, int(n_timesteps / 4))  # arrange stop is excluding

        os.makedirs(log_folder + '/learned_policy', exist_ok=True)
        os.makedirs(log_folder + '/saved_data', exist_ok=True)
        os.makedirs(log_folder + '/point_clouds', exist_ok=True)

    def callback(self, _locals, _globals):
        observation = self.convert_obs_to_dict(_locals['new_obs'])
        ep_infos = _locals['info']
        ag_points = ep_infos['achieved_goal']
        dg_points = ep_infos['desired_goal']
        errors = ep_infos['error']

        self.ag_points.extend(ag_points * 1000)
        self.dg_points.extend(dg_points * 1000)
        self.errors.append(errors * 1000)

        self.training_step = _locals['total_steps']

        # update goal tolerance if needed
        # Updated code
        _locals['self'].env.env.update_goal_tolerance(self.training_step)
        # Update rank
        if self.rank is not None:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0

        if _locals['total_steps'] in self.save_intervals:
            if rank == 0:
                self.convert_to_np_arrays()
                self.save_np_arrays(_locals['total_steps'])
                # self.save_point_cloud() # can do this in matlab
                self.clear_np_arrays()
                # Save model at intervals
                _locals['self'].save(self.log_folder + '/learned_policy/' + str(self.training_step) + '_saved_model.pkl')

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

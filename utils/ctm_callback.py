import os
import numpy as np

from pypcd import pypcd
from mpi4py import MPI
import h5py


# TODO: Add in goal tolerance class to update goal tolerance (sampling, decay or otherwise)
class CtmCallback(object):
    def __init__(self, log_folder, n_timesteps, goal_tolerance_parameters):
        self.log_folder = log_folder
        self.n_timesteps = n_timesteps
        self.num_workers = MPI.COMM_WORLD.Get_size()
        self.goal_tolerance_parameters = goal_tolerance_parameters

        self.ag_points = np.zeros([int(n_timesteps / 4) * self.num_workers, 3], dtype=float)
        self.errors = np.zeros(int(n_timesteps / 4) * self.num_workers, dtype=float)
        self.q_values = np.zeros(int(n_timesteps / 4) * self.num_workers, dtype=float)

        self.training_step = 0
        self.save_step = 0
        self.save_intervals = np.arange(0, n_timesteps + 1, int(n_timesteps / 4))  # arange stop is excluding

        os.makedirs(log_folder + '/learned_policy', exist_ok=True)
        os.makedirs(log_folder + '/saved_data', exist_ok=True)
        os.makedirs(log_folder + '/point_clouds', exist_ok=True)

    def callback(self, _locals, _globals):
        observation = _locals['self'].env.convert_obs_to_dict(_locals['new_obs'])
        print("saving intervals: ", self.save_intervals)
        print("save_step: ", self.save_step)
        print("training_step: ", self.training_step)
        self.ag_points[self.save_step, :] = observation['achieved_goal'] * 1000
        self.errors[self.save_step] = _locals['info']['error'] * 1000
        self.q_values[self.save_step] = _locals['q_value']

        rank = MPI.COMM_WORLD.Get_rank()
        self.training_step = _locals['total_steps']
        self.save_step += 1

        # update goal tolerance if needed
        if self.training_step in self.save_intervals:
            if rank == 0:
                self.save_np_arrays()
                self.save_point_cloud()
                self.clear_np_arrays()
                # Save model at intervals
                _locals['self'].save(self.log_folder + '/learned_policy/' + str(self.training_step) + '_saved_model.pkl')
            self.save_step = 0

    def save_np_arrays(self):
        hf = h5py.File(self.log_folder + '/saved_data/' + 'data_' + str(self.training_step) + '.h5', 'w')
        hf.create_dataset('achieved_goals', data=self.ag_points[:self.save_step, :])
        hf.create_dataset('errors', data=self.errors[:self.save_step])
        hf.create_dataset('q_values', data=self.q_values[:self.save_step])
        hf.close()

    def clear_np_arrays(self):
        self.ag_points = np.zeros([int(self.n_timesteps / 4) * self.num_workers, 3], dtype=float)
        self.errors = np.zeros(int(self.n_timesteps / 4) * self.num_workers, dtype=float)
        self.q_values = np.zeros(int(self.n_timesteps / 4) * self.num_workers, dtype=float)

    def save_point_cloud(self):
        ag_points = self.ag_points[:self.save_step, :]
        error_values = self.errors[:self.save_step]
        q_val_values = self.q_values[:self.save_step]

        assert not np.isnan(ag_points).any()
        assert not np.isnan(q_val_values).any()
        assert not np.isnan(error_values).any()

        q_pcl_points = np.ndarray((ag_points.shape[0], 4), dtype=np.float32)
        error_pcl_points = np.ndarray((ag_points.shape[0], 4), dtype=np.float32)
        q_pcl_points[:, :3] = ag_points
        error_pcl_points[:, :3] = ag_points

        error_rgb = pypcd.encode_rgb_for_pcl(self.vector_to_rgb(error_values))
        q_val_rgb = pypcd.encode_rgb_for_pcl(self.vector_to_rgb(q_val_values))
        error_pcl_points[:, -1] = error_rgb
        q_pcl_points[:, -1] = q_val_rgb

        q_value_cloud = pypcd.make_xyz_rgb_point_cloud(q_pcl_points)
        q_value_cloud.save_pcd(self.log_folder + "/point_clouds" + "/q_value_" + str(self.training_step) + ".pcd",
                               compression='binary')
        error_cloud = pypcd.make_xyz_rgb_point_cloud(error_pcl_points)
        error_cloud.save_pcd(self.log_folder + "/point_clouds" + "/error_" + str(self.training_step) + ".pcd", compression='binary')

    @staticmethod
    # Input: Vector of scalar values.
    # Output: ndarray of rgb value (int8) per scalar value
    def vector_to_rgb(input_scalar):
        minimum, maximum = np.min(input_scalar), np.max(input_scalar)
        ratio = 2 * (input_scalar - minimum) / (maximum - minimum)
        b = np.maximum(0, 255 * (1 - ratio)).astype(int)
        r = np.maximum(0, 255 * (ratio - 1)).astype(int)
        g = 255 - b - r
        rgb_values = np.empty((np.size(input_scalar), 3), dtype=np.uint8)
        rgb_values[:, 0] = r
        rgb_values[:, 1] = g
        rgb_values[:, 2] = b
        return rgb_values

    @staticmethod
    def _value_to_int8_rgba(value, minimum, maximum):
        minimum, maximum = float(minimum), float(maximum)
        ratio = 2 * (value - minimum) / (maximum - minimum)
        b = int(max(0, 255 * (1 - ratio)))
        r = int(max(0, 255 * (ratio - 1)))
        g = 255 - b - r
        rgb = np.array([r, g, b], dtype=np.uint8)
        # rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
        return np.reshape(rgb, [1, 3])


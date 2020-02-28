import struct
import numpy as np

from pypcd import pypcd
import time
from mpi4py import MPI
import h5py

import os
from functools import reduce

MM_TO_M = 1000


# Goal of this class to visualize concentric tube robot training in rviz.
# Publish joint states during training and / or evaluation.
# Publish training point cloud (reached cartesian point and associated error / q-value)
# Publish a voxel grid of training locations
class CallbackVisualizer(object):
    def __init__(self, log_folder, ros_flag, variable_goal_tolerance=True, exp_id=None):
        self._locals = None
        self._globals = None
        self._log_folder = log_folder
        self._ros_flag = ros_flag
        self._variable_goal_tolerance = variable_goal_tolerance
        if exp_id == 1:
            self.goal_tolerance_function = 'decay'
        elif exp_id == 2:
            self.goal_tolerance_function = 'linear'
        elif exp_id == 3:
            self.goal_tolerance_function = 'constant'
        else:
            print('Incorrect experiment id selected.')
        if self._ros_flag:
            import rospy
            import std_msgs.msg
            import sensor_msgs.point_cloud2 as pcl2
            from sensor_msgs.msg import PointCloud2, PointField
            self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                           PointField('y', 4, PointField.FLOAT32, 1),
                           PointField('z', 8, PointField.FLOAT32, 1),
                           PointField('rgba', 12, PointField.UINT32, 1)]
            self.q_value_pcl_pub = rospy.Publisher('ctm/q_value_pcl', PointCloud2, queue_size=10)
            self.error_pcl_pub = rospy.Publisher('ctm/error_pcl', PointCloud2, queue_size=10)

        self.ag_points = np.empty([600000 * 19, 3], dtype=float)
        self.errors = np.empty(600000 * 19, dtype=float)
        self.q_values = np.empty(600000 * 19, dtype=float)

        self.q_value_pcl = None
        self.error_pcl = None

        self.current_step = 0
        self.local_step = 0
        self.save_pcd_model_intervals = [2.5e5, 5e5, 7.5e5, 999995, 1e6]

        if self._variable_goal_tolerance:
            # Variable reward goal tolerance
            self.final_goal_tol = 0.0001
            self.initial_goal_tol = 0.050
            self.n_timesteps = 1e6

            if self.goal_tolerance_function == 'decay':
                self.a = self.initial_goal_tol
                self.r = 1 - np.power((self.final_goal_tol / self.initial_goal_tol), 1 / 2e6)
            elif self.goal_tolerance_function == 'linear':
                self.a = (self.final_goal_tol - self.initial_goal_tol) / self.n_timesteps
                self.b = self.initial_goal_tol
            else:
                self.goal_tolerance_function = 'constant'

    def callback(self, _locals, _globals):
        self._locals = _locals
        self._globals = _globals
        observation = _locals['self'].env.convert_obs_to_dict(_locals['new_obs'])
        ag = observation['achieved_goal'] * MM_TO_M
        self.ag_points[self.local_step, :] = ag
        error = _locals['info']['error']
        self.errors[self.local_step] = error
        q_val = _locals['q_value']
        self.q_values[self.local_step] = q_val
        rank = MPI.COMM_WORLD.Get_rank()

        self.current_step = _locals['total_steps'] - 1
        self.local_step += 1

        if self._variable_goal_tolerance:
            self._update_goal_tolerance()
        if self.current_step in self.save_pcd_model_intervals and rank == 0:
            self.save_arrays()
            # self.save_point_clouds()
            if self._ros_flag:
                self.publish_point_clouds()
            # Save model periodically
            _locals['self'].save(self._log_folder + '/' + 'temp_saved_model.pkl')
            # Clear array
            self.ag_points = np.empty([300000 * 19, 3], dtype=float)
            self.errors = np.empty(300000 * 19, dtype=float)
            self.q_values = np.empty(300000 * 19, dtype=float)
            self.local_step = 0

    def save_arrays(self):
        hf = h5py.File(self._log_folder + '/' + 'data_' + str(self.current_step) + '.h5', 'w')
        hf.create_dataset('achieved_goals', data=self.ag_points[0:self.local_step, :])
        hf.create_dataset('errors', data=self.errors[0:self.local_step])
        hf.create_dataset('q_values', data=self.q_values[0:self.local_step])
        hf.close()

    def save_point_clouds(self):
        ag_points = self.ag_points[0:self.local_step, :]
        error_values = self.errors[0:self.local_step]
        q_val_values = self.q_values[0:self.local_step]

        q_pcl_points = np.ndarray((ag_points.shape[0], 4), dtype=np.float32)
        error_pcl_points = np.ndarray((ag_points.shape[0], 4), dtype=np.float32)
        q_pcl_points[:, 0:3] = ag_points
        error_pcl_points[:, 0:3] = ag_points
        q_rgb_values = np.array([])
        error_rgb_values = np.array([])
        for val in q_val_values:
            q_rgb = pypcd.encode_rgb_for_pcl(self._value_to_int8_rgba(val, np.min(q_val_values), np.max(q_val_values)))
            q_rgb_values = np.append(q_rgb_values, q_rgb)
            error_rgb = pypcd.encode_rgb_for_pcl(self._value_to_int8_rgba(val, np.min(error_values), np.max(error_values)))
            error_rgb_values = np.append(error_rgb_values, error_rgb)

        q_pcl_points[:, 3] = q_rgb_values
        error_pcl_points[:, 3] = error_rgb_values

        q_value_cloud = pypcd.make_xyz_rgb_point_cloud(q_pcl_points)
        q_value_cloud.save_pcd(self._log_folder + "/q_value_" + str(self.current_step + 1) + ".pcd", compression='binary')
        error_cloud = pypcd.make_xyz_rgb_point_cloud(error_pcl_points)
        error_cloud.save_pcd(self._log_folder + "/error_" + str(self.current_step + 1) + ".pcd", compression='binary')

        if self._ros_flag:
            import rospy
            import std_msgs.msg
            import sensor_msgs.point_cloud2 as pcl2
            from sensor_msgs.msg import PointCloud2, PointField
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "world"
            # Normalize the rgb values before publishing.
            self.q_value_pcl = pcl2.create_cloud(header, self.fields, q_pcl_points.transpose().tolist())
            self.error_pcl = pcl2.create_cloud(header, self.fields, error_pcl_points.transpose().tolist())

    def publish_point_clouds(self):
        self.q_value_pcl_pub.publish(self.q_value_pcl)
        self.error_pcl_pub.publish(self.error_pcl)

        # How to read and publish read point cloud
        # test_cloud = read_pcd('test.pcd')
        # test_cloud.header.stamp = rospy.Time.now()
        # test_cloud.header.frame_id = "world"
        # self.error_pcl_pub.publish(test_cloud)

    @staticmethod
    def _value_to_int8_rgba(value, minimum, maximum):
        minimum, maximum = float(minimum), float(maximum)
        ratio = 2 * (value - minimum) / (maximum - minimum)
        b = int(max(0, 255 * (1 - ratio)))
        r = int(max(0, 255 * (ratio - 1)))
        g = 255 - b - r
        rgb = np.array([r, g, b], dtype=np.uint8)
        #rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
        return np.reshape(rgb, [1, 3])

    def _update_goal_tolerance(self):
        if self.goal_tolerance_function == 'decay':
            goal_tol_new = self.a * np.power(1 - self.r, self.current_step)
        elif self.goal_tolerance_function == 'linear':
            goal_tol_new = self.a * self.current_step + self.b
        else:
            goal_tol_new = self.final_goal_tol
        self._locals['self'].env.env.update_goal_tolerance(goal_tol_new)

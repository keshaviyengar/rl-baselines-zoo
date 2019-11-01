import struct
import numpy as np

from pypcd import pypcd

import os
from functools import reduce

MM_TO_M = 1000


# Goal of this class to visualize concentric tube robot training in rviz.
# Publish joint states during training and / or evaluation.
# Publish training point cloud (reached cartesian point and associated error / q-value)
# Publish a voxel grid of training locations
class CallbackVisualizer(object):
    def __init__(self, log_folder, ros_flag):
        self._locals = None
        self._globals = None
        self._log_folder = log_folder
        self._ros_flag = ros_flag
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

        self.pcl_points = np.array([])
        self.ag_points = np.array([])
        self.errors = np.array([])
        self.q_values = np.array([])

        self.q_value_pcl = None
        self.error_pcl = None

    def callback(self, _locals, _globals):
        self._locals = _locals
        self._globals = _globals
        observation = _locals['self'].env.convert_obs_to_dict(_locals['new_obs'])
        ag = observation['achieved_goal'] * MM_TO_M
        self.ag_points = np.append(self.ag_points, ag)
        error = _locals['info']['error']
        self.errors = np.append(self.errors, error)
        q_val = _locals['q_value']
        self.q_values = np.append(self.q_values, q_val)

        if _locals['total_steps'] % 10000 == 0:
            self.save_point_clouds()
            if self._ros_flag:
                self.publish_point_clouds()

    def save_point_clouds(self):
        ag_points = self.ag_points.reshape(-1, 3)
        q_pcl_points = np.ndarray((ag_points.shape[0], 4), dtype=np.float32)
        error_pcl_points = np.ndarray((ag_points.shape[0], 4), dtype=np.float32)
        q_pcl_points[:, 0:3] = ag_points
        error_pcl_points[:, 0:3] = ag_points
        q_rgb_values = np.array([])
        error_rgb_values = np.array([])
        for val in self.q_values:
            q_rgb = pypcd.encode_rgb_for_pcl(self._value_to_int8_rgba(val, np.min(self.q_values), np.max(self.q_values)))
            q_rgb_values = np.append(q_rgb_values, q_rgb)
            error_rgb = pypcd.encode_rgb_for_pcl(self._value_to_int8_rgba(val, np.min(self.errors), np.max(self.errors)))
            error_rgb_values = np.append(error_rgb_values, error_rgb)

        q_pcl_points[:, 3] = q_rgb_values
        error_pcl_points[:, 3] = error_rgb_values

        #q_value_cloud = pypcd.make_xyz_rgb_point_cloud(q_pcl_points, metadata={'width': q_pcl_points.shape[1], 'points': q_pcl_points.shape[1]})
        q_value_cloud = pypcd.make_xyz_rgb_point_cloud(q_pcl_points)
        q_value_cloud.save_pcd(self._log_folder + "/q_value.pcd", compression='binary')
        #error_cloud = pypcd.make_xyz_rgb_point_cloud(error_pcl_points, metadata={'width': error_pcl_points.shape[1], 'points': error_pcl_points.shape[1]})
        error_cloud = pypcd.make_xyz_rgb_point_cloud(error_pcl_points)
        error_cloud.save_pcd(self._log_folder + "/error.pcd", compression='binary')

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

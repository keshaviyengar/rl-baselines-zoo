import rospy
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField

import struct
import numpy as np


MM_TO_M = 1000


# Goal of this class to visualize concentric tube robot training in rviz.
# Publish joint states during training and / or evaluation.
# Publish training point cloud (reached cartesian point and associated error / q-value)
# Publish a voxel grid of training locations
class CallbackVisualizer(object):
    def __init__(self):
        self._locals = None
        self._globals = None
        self.pcl_points = np.array([])
        self.ag_points = np.array([])
        self.errors = np.array([])
        self.q_values = np.array([])
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                       PointField('y', 4, PointField.FLOAT32, 1),
                       PointField('z', 8, PointField.FLOAT32, 1),
                       PointField('rgba', 12, PointField.UINT32, 1)]

        self.ag_pcl_pub = rospy.Publisher('ctm/achieved_goal_pcl', PointCloud2, queue_size=10)
        self.error_pcl_pub = rospy.Publisher('ctm/error_pcl', PointCloud2, queue_size=10)
        rospy.init_node("pcl_visualizer", anonymous=True)

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

        if _locals['total_steps'] % 500 == 0:
            self.publish_achieved_goal_pcl()

    def publish_achieved_goal_pcl(self):
        array_to_rgba_array = np.vectorize(self._value_to_int32_rgba)

        normalized_rgba = np.squeeze(array_to_rgba_array(self.q_values, np.min(self.q_values), np.max(self.q_values)))
        ag_points = self.ag_points.reshape(-1, 3).transpose()
        ag_pcl_points = np.ndarray((4, ag_points.shape[1]), dtype=object)
        ag_pcl_points[0:3, 0:] = ag_points
        ag_pcl_points[3, :] = normalized_rgba

        normalized_error = np.squeeze(array_to_rgba_array(self.errors, np.min(self.errors), np.max(self.errors)))
        error_pcl_points = np.ndarray((4, ag_points.shape[1]), dtype=object)
        error_pcl_points[0:3, 0:] = ag_points
        error_pcl_points[3, :] = normalized_error

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"
        #point_cloud = pcl2.create_cloud_xyz32(header, self.ag_points)
        # Normalize the rgb values before publishing.
        ag_point_cloud = pcl2.create_cloud(header, self.fields, ag_pcl_points.transpose().tolist())
        error_point_cloud = pcl2.create_cloud(header, self.fields, error_pcl_points.transpose().tolist())
        self.ag_pcl_pub.publish(ag_point_cloud)
        self.error_pcl_pub.publish(error_point_cloud)

    @staticmethod
    def _value_to_int32_rgba(value, minimum, maximum):
        minimum, maximum = float(minimum), float(maximum)
        ratio = 2 * (value - minimum) / (maximum - minimum)
        b = int(max(0, 255 * (1 - ratio)))
        r = int(max(0, 255 * (ratio - 1)))
        g = 255 - b - r
        a = 255
        rgba = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        return rgba


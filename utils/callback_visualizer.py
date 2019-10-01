import rospy
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2


MM_TO_M = 1000


# Goal of this class to visualize concentric tube robot training in rviz.
# Publish joint states during training and / or evaluation.
# Publish training point cloud (reached cartesian point and associated error / q-value)
# Publish a voxel grid of training locations
class CallbackVisualizer(object):
    def __init__(self):
        self._locals = None
        self._globals = None
        self.ag_points = []
        self.errors = []

        self.ag_pcl_pub = rospy.Publisher('ctm/achieved_goal_pcl', PointCloud2, queue_size=10)
        rospy.init_node("pcl_visualizer", anonymous=True)

    def callback(self, _locals, _globals):
        self._locals = _locals
        self._globals = _globals
        observation = _locals['self'].env.convert_obs_to_dict(_locals['new_obs'])
        ag = observation['achieved_goal']
        error = _locals['info']['error']
        self.ag_points.append(ag * MM_TO_M)
        self.errors.append(error)

        if _locals['total_steps'] % 1000 == 0:
            self.publish_achieved_goal_pcl()

    def publish_achieved_goal_pcl(self):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "base_link"
        point_cloud = pcl2.create_cloud_xyz32(header, self.ag_points)
        self.ag_pcl_pub.publish(point_cloud)

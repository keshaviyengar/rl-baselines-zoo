import rospy
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Bool

# This script creates a square trajectory for a robot to follow.
# Will output errors as well.


class SquareTrajectory(object):
    def __init__(self):
        rospy.init_node("square_trajectory_generator")
        self.trajectory_pub = rospy.Publisher("desired_goal", Pose, queue_size=10)
        self.trajectory_finish_pub = rospy.Publisher("trajectory_finish", Bool, queue_size=10)
        self._current_pose = Pose()

        # Subscribe to current end effector position / pose
        self._current_pose_sub = rospy.Subscriber("/ctm/achieved_goal", Point, self._current_pose_callback)

        # Create a timer to update the desired trajectory
        rospy.Timer(rospy.Duration(0.01), self._trajectory_callback)

        # Second timer for how long to move in axis before moving to next
        rospy.Timer(rospy.Duration(5.0), self._change_direction)

        self._turn_count = 0
        self._done_trajectory = False
        # For now set initial current pose as 0
        self._desired_pose = Pose()
        self._desired_pose.position.x = 10  # 10 for 3 and 4 tube, 30 for 2 tube
        self._desired_pose.position.y = 10  # 0 for 3 and 4 tube, 30 for 2 tube
        self._desired_pose.position.z = 120  # 120 for 3 and 4 tube, 100 for 2 tube
        self._desired_pose.orientation.x = 0
        self._desired_pose.orientation.y = 0
        self._desired_pose.orientation.z = 0
        self._desired_pose.orientation.w = 1

        self.speed = 2

        self.prev_time = rospy.get_time()

    # This callback changes the direction by 90 degrees, to make the square.
    def _change_direction(self, event):
        self._turn_count += 1
        if self._turn_count == 4:
            print("Finished, reached last turn.")
            self._done_trajectory = True

    def _trajectory_callback(self, event):
        # Compute current difference in time from last callback
        current_time = rospy.get_time()
        delta_t = current_time - self.prev_time
        self.prev_time = current_time
        if not self._done_trajectory:
            if self._turn_count == 0:
                # Negative x
                self._desired_pose.position.x -= self.speed * delta_t
            if self._turn_count == 1:
                # Positive y
                self._desired_pose.position.y += self.speed * delta_t
            if self._turn_count == 2:
                # Positive x
                self._desired_pose.position.x += self.speed * delta_t
            if self._turn_count == 3:
                # Negative y
                self._desired_pose.position.y -= self.speed * delta_t
            # Publish new pose
            self.trajectory_pub.publish(self._desired_pose)
        else:
            print("Trajectory is complete.")
            self.trajectory_finish_pub.publish(True)

    # Update current pose for error calculation
    def _current_pose_callback(self, msg):
        self._current_pose.position.x = msg.x
        self._current_pose.position.y = msg.y
        self._current_pose.position.z = msg.z


if __name__ == '__main__':
    square_trajectory = SquareTrajectory()
    rospy.spin()

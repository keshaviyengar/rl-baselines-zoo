import rospy
from geometry_msgs.msg import Pose

# This script creates a square trajectory for a robot to follow.
# Will output errors as well.


class SquareTrajectory(object):
    def __init__(self):
        rospy.init_node("square_trajectory_generator")
        self.trajectory_pub = rospy.Publisher("desired_goal", Pose, queue_size=10)
        self._current_pose = Pose()

        # Subscribe to current end effector position / pose
        self._current_pose_sb = rospy.Subscriber("current_pose", Pose, self._current_pose_callback)

        # Create a timer to update the desired trajectory
        rospy.Timer(rospy.Duration(0.01), self._trajectory_callback)

        # Second timer for how long to move in axis before moving to next
        rospy.Timer(rospy.Duration(5.0), self._change_direction)

        self._turn_count = 0
        self._done_trajectory = False
        # For now set initial current pose as first desired pose point
        self._desired_pose = self._current_pose

        self.speed = 1

    # This callback changes the direction by 90 degrees, to make the square.
    def _change_direction(self):
        self._turn_count += 1
        if self._turn_count == 4:
            print("Finished, reached last turn.")
            self._done_trajectory = False

    def _trajectory_callback(self):
        if not self._done_trajectory:
            if self._turn_count == 0:
                # Negative x
                self._desired_pose.position.x -= self.speed * 0.01
            if self._turn_count == 1:
                # Positive y
                self._desired_pose.position.y += self.speed * 0.01
            if self._turn_count == 2:
                # Positive x
                self._desired_pose.position.x += self.speed * 0.01
            if self._turn_count == 3:
                # Negative y
                self._desired_pose.position.y -= self.speed * 0.01
        else:
            print("Trajectory is complete.")

    # Update current pose for error calculation
    def _current_pose_callback(self, msg):
        self._current_pose = msg.pose


if __name__ == '__main__':
    square_trajectory = SquareTrajectory()

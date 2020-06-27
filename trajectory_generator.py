import rospy
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Bool

import numpy as np
import os
# This script creates a square trajectory for a robot to follow.
# Will output errors as well.


class CircleTrajectory(object):
    def __init__(self, x_offset, y_offset, z_height, radius, theta_step):
        self.trajectory_pub = rospy.Publisher("desired_goal", Pose, queue_size=10)
        self.trajectory_finish_pub = rospy.Publisher("trajectory_finish", Bool, queue_size=10)
        self._current_pose = Pose()

        # Create a timer to update the desired trajectory
        self.trajectory_timer = rospy.Timer(rospy.Duration(0.01), self._trajectory_callback)

        self.traj_finish = False
        # For now set initial current pose as 0
        self._desired_pose = Pose()
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.radius = radius
        self.thetas = np.arange(0, 2 * np.pi, np.deg2rad(theta_step))
        self.thetas_counter = 0
        self._desired_pose.position.x = self.x_offset + self.radius * np.cos(self.thetas[self.thetas_counter])
        self._desired_pose.position.y = self.y_offset + self.radius * np.sin(self.thetas[self.thetas_counter])
        self._desired_pose.position.z = z_height
        self._desired_pose.orientation.x = 0
        self._desired_pose.orientation.y = 0
        self._desired_pose.orientation.z = 0
        self._desired_pose.orientation.w = 1

        self.speed = 1

    def _trajectory_callback(self, event):
        self.thetas_counter += 1
        if self.thetas_counter == self.thetas.size - 1:
            self.traj_finish = True
            print("Trajectory is complete.")
            self.trajectory_finish_pub.publish(True)
            self.trajectory_timer.shutdown()

        if not self.traj_finish:
            self._desired_pose.position.x = self.x_offset + self.radius * np.cos(self.thetas[self.thetas_counter])
            self._desired_pose.position.y = self.y_offset + self.radius * np.sin(self.thetas[self.thetas_counter])
            # Publish new pose
            self.trajectory_pub.publish(self._desired_pose)


class TriangleTrajectory(object):
    def __init__(self, point_a, point_b, point_c, z_height):
        self.trajectory_pub = rospy.Publisher("desired_goal", Pose, queue_size=10)
        self.trajectory_finish_pub = rospy.Publisher("trajectory_finish", Bool, queue_size=10)
        self._current_pose = Pose()


        # Second timer for how long to move in axis before moving to next
        # self.change_direction_timer = rospy.Timer(rospy.Duration(5.0), self._change_direction)

        # Specify three points to reach to create the triangle
        self.points = np.array([point_a, point_b, point_c])

        self._turn_count = 0
        self.del_vector = [(self.points[1][0] - self.points[0][0]), (self.points[1][1] - self.points[0][1])]

        self._done_trajectory = False

        self._desired_pose = Pose()
        self._desired_pose.position.x = point_a[0]
        self._desired_pose.position.y = point_a[1]
        self._desired_pose.position.z = z_height
        self._desired_pose.orientation.x = 0
        self._desired_pose.orientation.y = 0
        self._desired_pose.orientation.z = 0
        self._desired_pose.orientation.w = 1

        # Publish initial point and sleep to initialize
        for _ in range(10):
            self.trajectory_pub.publish(self._desired_pose)
            rospy.sleep(0.1)

        self.prev_time = rospy.get_time()
        self.traj_finish = False

        # Create a timer to update the desired trajectory
        self.trajectory_timer = rospy.Timer(rospy.Duration(0.01), self._trajectory_callback)

    # This callback changes the direction by 90 degrees, to make the square.
    def _change_direction(self):
        if self._turn_count == 0:
            if np.linalg.norm(self.points[self._turn_count + 1] - np.array(
                    [self._desired_pose.position.x, self._desired_pose.position.y])) < 0.5:
                self._turn_count += 1
            self.del_vector = [(self.points[1][0] - self.points[0][0]),
                               (self.points[1][1] - self.points[0][1])]

        if self._turn_count == 1:
            if np.linalg.norm(self.points[self._turn_count + 1] - np.array(
                    [self._desired_pose.position.x, self._desired_pose.position.y])) < 0.5:
                self._turn_count += 1
            self.del_vector = [(self.points[2][0] - self.points[1][0]),
                               (self.points[2][1] - self.points[1][1])]
        if self._turn_count == 2:
            if np.linalg.norm(self.points[0] - np.array(
                    [self._desired_pose.position.x, self._desired_pose.position.y])) < 0.5:
                self._turn_count += 1
            self.del_vector = [(self.points[0][0] - self.points[2][0]),
                               (self.points[0][1] - self.points[2][1])]
        if self._turn_count == 3:
            print("Trajectory is complete.")
            self.traj_finish = True
            self.trajectory_finish_pub.publish(True)
            self.trajectory_timer.shutdown()
            # self.change_direction_timer.shutdown()

    def _trajectory_callback(self, event):
        # Compute current difference in time from last callback
        if not self.traj_finish:
            current_time = rospy.get_time()
            delta_t = current_time - self.prev_time
            self.prev_time = current_time

            self._change_direction()

            self._desired_pose.position.x += self.del_vector[0] * delta_t
            self._desired_pose.position.y += self.del_vector[1] * delta_t
            self.trajectory_pub.publish(self._desired_pose)


class SquareTrajectory2(object):
    def __init__(self, point_a, point_b, point_c, point_d, z_height):
        self.trajectory_pub = rospy.Publisher("desired_goal", Pose, queue_size=10)
        self.trajectory_finish_pub = rospy.Publisher("trajectory_finish", Bool, queue_size=10)
        self._current_pose = Pose()

        self.points = [point_a, point_b, point_c, point_d]

        self._turn_count = 0
        self.del_vector = [(self.points[1][0] - self.points[0][0]), (self.points[1][1] - self.points[0][1])]

        # For now set initial current pose as 0
        self._desired_pose = Pose()
        self._desired_pose.position.x = point_a[0]
        self._desired_pose.position.y = point_a[1]
        self._desired_pose.position.z = z_height
        self._desired_pose.orientation.x = 0
        self._desired_pose.orientation.y = 0
        self._desired_pose.orientation.z = 0
        self._desired_pose.orientation.w = 1

        # Publish initial point and sleep to initialize
        for _ in range(10):
            self.trajectory_pub.publish(self._desired_pose)
            rospy.sleep(0.1)

        self.prev_time = rospy.get_time()
        self.traj_finish = False

        # Create a timer to update the desired trajectory
        self.trajectory_timer = rospy.Timer(rospy.Duration(0.01), self._trajectory_callback)

    # This callback changes the direction by 90 degrees, to make the square.
    def _change_direction(self):
        if self._turn_count == 0:
            if np.linalg.norm(self.points[self._turn_count + 1] - np.array(
                    [self._desired_pose.position.x, self._desired_pose.position.y])) < 0.5:
                self._turn_count += 1
            self.del_vector = [(self.points[1][0] - self.points[0][0]),
                               (self.points[1][1] - self.points[0][1])]

        if self._turn_count == 1:
            if np.linalg.norm(self.points[self._turn_count + 1] - np.array(
                    [self._desired_pose.position.x, self._desired_pose.position.y])) < 0.5:
                self._turn_count += 1
            self.del_vector = [(self.points[2][0] - self.points[1][0]),
                               (self.points[2][1] - self.points[1][1])]
        if self._turn_count == 2:
            if np.linalg.norm(self.points[self._turn_count + 1] - np.array(
                    [self._desired_pose.position.x, self._desired_pose.position.y])) < 0.5:
                self._turn_count += 1
            self.del_vector = [(self.points[3][0] - self.points[2][0]),
                               (self.points[3][1] - self.points[2][1])]
        if self._turn_count == 3:
            if np.linalg.norm(self.points[0] - np.array(
                    [self._desired_pose.position.x, self._desired_pose.position.y])) < 0.5:
                self._turn_count += 1
            self.del_vector = [(self.points[0][0] - self.points[3][0]),
                               (self.points[0][1] - self.points[3][1])]
        if self._turn_count == 4:
            print("Trajectory is complete.")
            self.traj_finish = True
            self.trajectory_finish_pub.publish(True)
            self.trajectory_timer.shutdown()

    def _trajectory_callback(self, event):
        # Compute current difference in time from last callback
        if not self.traj_finish:
            current_time = rospy.get_time()
            delta_t = current_time - self.prev_time
            self.prev_time = current_time

            self._change_direction()

            self._desired_pose.position.x += self.del_vector[0] * delta_t
            self._desired_pose.position.y += self.del_vector[1] * delta_t
            self.trajectory_pub.publish(self._desired_pose)


if __name__ == '__main__':
    rospy.init_node("trajectory_generator")
    experiments = [7]

    for exp in experiments:
        x_offset = 5
        y_offset = 5
        if exp in [1, 2, 3, 4, 5]:
            z_height = 100
        elif exp in [6, 7, 8, 9, 10]:
            z_height = 100
        else:
            z_height = 125

        radius = 2.0
        theta_step = 0.5
        print("Circle trajectory")
        circle_trajectory = CircleTrajectory(x_offset, y_offset, z_height, radius, theta_step)
        while not circle_trajectory.traj_finish:
            if circle_trajectory.traj_finish:
                break

        # point_a = [20, 20]
        # point_b = [20, 30]
        # point_c = [30, 20]
        # point_a = [-5, 0]
        # point_b = [-10, -5]
        # point_c = [5, 0]
        # if exp in [1, 2, 3, 4, 5]:
        #     z_height = 100
        # elif exp in [6, 7, 8, 9, 10]:
        #     z_height = 125
        # else:
        #     z_height = 125
        # print("Triangle trajectory")
        # triangle_trajectory = TriangleTrajectory(point_a, point_b, point_c, z_height)
        # while not triangle_trajectory.traj_finish:
        #     pass

        # point_a = [5, 0]
        # point_b = [-5, 0]
        # point_c = [-5, -5]
        # point_d = [5, -5]
        # if exp in [1, 2, 3, 4, 5]:
        #     z_height = 100
        # elif exp in [6, 7, 8, 9, 10]:
        #     z_height = 125
        # else:
        #     z_height = 125
        # print("Square trajectory")
        # square_trajectory = SquareTrajectory2(point_a, point_b, point_c, point_d, z_height)
        # while not square_trajectory.traj_finish:
        #     pass


#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import os
import csv
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
import math


class PurePursuit(Node):
    """
    Pure Pursuit path tracking on the F1Tenth car.
    Subscribes to pose (sim or particle filter) and publishes drive commands.
    """

    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Declare parameters
        self.declare_parameter('waypoint_file', '')
        self.declare_parameter('lookahead_distance', 1.5)
        self.declare_parameter('lookahead_gain', 0.5)  # speed-dependent lookahead: L = lookahead_gain * v + min_lookahead
        self.declare_parameter('min_lookahead', 0.5)
        self.declare_parameter('max_lookahead', 3.0)
        self.declare_parameter('velocity', 1.5)
        self.declare_parameter('max_steering_angle', 0.4189)  # ~24 degrees
        self.declare_parameter('wheelbase', 0.3302)
        self.declare_parameter('use_odom', True)  # True for sim (odom), False for particle filter (PoseStamped)
        self.declare_parameter('speed_lookahead', False)  # Use speed-dependent lookahead

        # Get parameters
        waypoint_file = self.get_parameter('waypoint_file').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.lookahead_gain = self.get_parameter('lookahead_gain').value
        self.min_lookahead = self.get_parameter('min_lookahead').value
        self.max_lookahead = self.get_parameter('max_lookahead').value
        self.velocity = self.get_parameter('velocity').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.wheelbase = self.get_parameter('wheelbase').value
        self.use_odom = self.get_parameter('use_odom').value
        self.speed_lookahead = self.get_parameter('speed_lookahead').value

        # Load waypoints
        self.waypoints = self.load_waypoints(waypoint_file)
        if self.waypoints is None or len(self.waypoints) == 0:
            self.get_logger().error('No waypoints loaded! Please provide a valid waypoint file.')
            return

        self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints from {waypoint_file}')

        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.waypoint_viz_pub = self.create_publisher(MarkerArray, '/waypoints_viz', 10)
        self.goal_viz_pub = self.create_publisher(Marker, '/goal_waypoint_viz', 10)

        # Subscribers
        if self.use_odom:
            self.pose_sub = self.create_subscription(
                Odometry, '/ego_racecar/odom', self.odom_callback, 10)
            self.get_logger().info('Subscribing to /ego_racecar/odom (simulator mode)')
        else:
            self.pose_sub = self.create_subscription(
                PoseStamped, '/pf/viz/inferred_pose', self.pose_callback, 10)
            self.get_logger().info('Subscribing to /pf/viz/inferred_pose (particle filter mode)')

        # Publish waypoints visualization once with a timer
        self.viz_timer = self.create_timer(1.0, self.publish_waypoints_viz)

        self.get_logger().info('PurePursuit node initialized')

    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        """Convert quaternion to yaw angle."""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def load_waypoints(self, filepath):
        """Load waypoints from CSV file. Expected format: x, y [, theta, velocity, ...]"""
        if not filepath or not os.path.exists(filepath):
            self.get_logger().warn(f'Waypoint file not found: {filepath}')
            return None

        waypoints = []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    x = float(row[0])
                    y = float(row[1])
                    v = float(row[3]) if len(row) > 3 else self.velocity
                    waypoints.append([x, y, v])
                except (ValueError, IndexError):
                    continue

        return np.array(waypoints) if waypoints else None

    def odom_callback(self, odom_msg):
        """Handle Odometry messages (simulator ground truth)."""
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        quat = odom_msg.pose.pose.orientation
        yaw = self.quaternion_to_yaw(quat.x, quat.y, quat.z, quat.w)
        self.pursue(x, y, yaw)

    def pose_callback(self, pose_msg):
        """Handle PoseStamped messages (particle filter)."""
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        quat = pose_msg.pose.orientation
        yaw = self.quaternion_to_yaw(quat.x, quat.y, quat.z, quat.w)
        self.pursue(x, y, yaw)

    def pursue(self, x, y, yaw):
        """Core Pure Pursuit algorithm."""
        if self.waypoints is None or len(self.waypoints) == 0:
            return

        # Current position
        car_pos = np.array([x, y])

        # Find the nearest waypoint
        distances = np.linalg.norm(self.waypoints[:, :2] - car_pos, axis=1)
        nearest_idx = np.argmin(distances)

        # Determine lookahead distance
        if self.speed_lookahead:
            current_v = self.waypoints[nearest_idx, 2]
            L = np.clip(
                self.lookahead_gain * current_v + self.min_lookahead,
                self.min_lookahead, self.max_lookahead)
        else:
            L = self.lookahead_distance

        # Find the goal waypoint: first waypoint beyond lookahead distance
        # Search forward from nearest waypoint
        n = len(self.waypoints)
        goal_idx = nearest_idx
        for i in range(n):
            idx = (nearest_idx + i) % n
            d = np.linalg.norm(self.waypoints[idx, :2] - car_pos)
            if d >= L:
                goal_idx = idx
                break

        goal_x = self.waypoints[goal_idx, 0]
        goal_y = self.waypoints[goal_idx, 1]
        goal_v = self.waypoints[goal_idx, 2]

        # Transform goal point to vehicle frame
        dx = goal_x - x
        dy = goal_y - y
        # Rotate into vehicle frame (vehicle x is forward, y is left)
        goal_x_vehicle = dx * np.cos(yaw) + dy * np.sin(yaw)
        goal_y_vehicle = -dx * np.sin(yaw) + dy * np.cos(yaw)

        # Calculate curvature: gamma = 2 * |y| / L^2
        # Steering angle: delta = atan(gamma * wheelbase)
        # Sign of y determines turn direction
        L_actual = np.sqrt(goal_x_vehicle**2 + goal_y_vehicle**2)
        if L_actual < 0.001:
            return

        curvature = 2.0 * goal_y_vehicle / (L_actual**2)
        steering_angle = np.arctan(curvature * self.wheelbase)

        # Clip steering angle
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

        # Adjust velocity based on steering angle (slow down for sharp turns)
        abs_steer = abs(steering_angle)
        if abs_steer > 0.35:
            speed = min(goal_v, 0.5)
        elif abs_steer > 0.2:
            speed = min(goal_v, 1.0)
        else:
            speed = goal_v

        # Publish drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

        # Visualize goal waypoint
        self.publish_goal_viz(goal_x, goal_y)

    def publish_waypoints_viz(self):
        """Publish all waypoints as MarkerArray for RViz."""
        if self.waypoints is None:
            return

        marker_array = MarkerArray()
        for i, wp in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'waypoints'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = wp[0]
            marker.pose.position.y = wp[1]
            marker.pose.position.z = 0.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 0.8
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker_array.markers.append(marker)

        self.waypoint_viz_pub.publish(marker_array)

    def publish_goal_viz(self, goal_x, goal_y):
        """Publish the current goal waypoint as a marker."""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = goal_x
        marker.pose.position.y = goal_y
        marker.pose.position.z = 0.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        self.goal_viz_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)
    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
waypoint_logger.py

Records waypoints from the particle filter pose estimate.
Drive the car manually around the track, then Ctrl+C to save.

Usage:
  ros2 run pure_pursuit waypoint_logger.py --ros-args \
    -p output_file:=/home/nvidia/f1tenth_ws/src/lab-5-slam-and-pure-pursuit-team14/pure_pursuit/waypoints/waypoints_real.csv \
    -p min_distance:=0.2 \
    -p use_odom:=false
"""

import rclpy
from rclpy.node import Node

import csv
import os
import math
import numpy as np

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class WaypointLogger(Node):

    def __init__(self):
        super().__init__('waypoint_logger')

        # Parameters
        self.declare_parameter('output_file',
            '/home/nvidia/f1tenth_ws/src/lab-5-slam-and-pure-pursuit-team14/pure_pursuit/waypoints/waypoints_real.csv')
        self.declare_parameter('min_distance', 0.2)   # meters between waypoints
        self.declare_parameter('use_odom', False)      # False = particle filter, True = sim odom
        self.declare_parameter('recorded_velocity', 1.5)  # velocity tag written to each waypoint

        self.output_file      = self.get_parameter('output_file').value
        self.min_distance     = self.get_parameter('min_distance').value
        self.use_odom         = self.get_parameter('use_odom').value
        self.recorded_velocity = self.get_parameter('recorded_velocity').value

        # State
        self.waypoints = []           # list of [x, y, theta, v]
        self.last_pos  = None         # last recorded (x, y)
        self.total_distance = 0.0

        # Subscribe to pose source
        if self.use_odom:
            self.sub = self.create_subscription(
                Odometry,
                '/ego_racecar/odom',
                self.odom_callback,
                10)
            self.get_logger().info('Listening to /ego_racecar/odom (odom mode)')
        else:
            self.sub = self.create_subscription(
                PoseStamped,
                '/pf/viz/inferred_pose',
                self.pose_callback,
                10)
            self.get_logger().info('Listening to /pf/viz/inferred_pose (particle filter mode)')

        # Status timer — prints count every 2 seconds so you know it's alive
        self.timer = self.create_timer(2.0, self.status_callback)

        self.get_logger().info('=' * 50)
        self.get_logger().info('Waypoint Logger Started')
        self.get_logger().info(f'  Output file  : {self.output_file}')
        self.get_logger().info(f'  Min distance : {self.min_distance} m')
        self.get_logger().info(f'  Drive the car around the track, then Ctrl+C to save.')
        self.get_logger().info('=' * 50)

    # ------------------------------------------------------------------
    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        p = msg.pose.pose
        yaw = self.quaternion_to_yaw(
            p.orientation.x, p.orientation.y,
            p.orientation.z, p.orientation.w)
        self.record(p.position.x, p.position.y, yaw)

    def pose_callback(self, msg):
        p = msg.pose
        yaw = self.quaternion_to_yaw(
            p.orientation.x, p.orientation.y,
            p.orientation.z, p.orientation.w)
        self.record(p.position.x, p.position.y, yaw)

    # ------------------------------------------------------------------
    def record(self, x, y, theta):
        # Skip if not moved far enough
        if self.last_pos is not None:
            dx = x - self.last_pos[0]
            dy = y - self.last_pos[1]
            dist = math.hypot(dx, dy)
            if dist < self.min_distance:
                return
            self.total_distance += dist
        
        self.last_pos = (x, y)
        self.waypoints.append([
            round(x, 4),
            round(y, 4),
            round(theta, 4),
            self.recorded_velocity
        ])

        # Print every 10 waypoints so terminal doesn't flood
        if len(self.waypoints) % 10 == 0:
            self.get_logger().info(
                f'Recorded {len(self.waypoints)} waypoints | '
                f'Distance: {self.total_distance:.1f}m | '
                f'Last: ({x:.2f}, {y:.2f})')

    # ------------------------------------------------------------------
    def status_callback(self):
        if len(self.waypoints) == 0:
            self.get_logger().warn(
                'No waypoints recorded yet — is the pose topic publishing? '
                'Check: ros2 topic hz /pf/viz/inferred_pose')
        else:
            self.get_logger().info(
                f'[STATUS] {len(self.waypoints)} waypoints | '
                f'{self.total_distance:.1f}m driven')

    # ------------------------------------------------------------------
    def save(self):
        if not self.waypoints:
            self.get_logger().error('No waypoints to save!')
            return

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.waypoints)

        self.get_logger().info('=' * 50)
        self.get_logger().info(f'Saved {len(self.waypoints)} waypoints to:')
        self.get_logger().info(f'  {self.output_file}')
        self.get_logger().info(f'  Total distance : {self.total_distance:.1f} m')
        self.get_logger().info(f'  First point    : {self.waypoints[0]}')
        self.get_logger().info(f'  Last point     : {self.waypoints[-1]}')
        self.get_logger().info('=' * 50)

        # Quick sanity check — warn if loop isn't closed
        first = np.array(self.waypoints[0][:2])
        last  = np.array(self.waypoints[-1][:2])
        gap   = np.linalg.norm(first - last)
        if gap > 1.0:
            self.get_logger().warn(
                f'Loop gap is {gap:.2f}m — first and last points are far apart. '
                f'Did you complete the full lap?')
        else:
            self.get_logger().info(f'Loop closed nicely (gap={gap:.2f}m)')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
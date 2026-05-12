#!/usr/bin/env python3
"""Subscribe to /move_base_simple/goal and /clicked_point, visualize and save waypoints."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import csv
import math


class WaypointClicker(Node):
    def __init__(self):
        super().__init__('waypoint_clicker')
        self.declare_parameter('output_file', 'waypoints.csv')
        self.declare_parameter('default_velocity', 1.0)

        self.output_file = self.get_parameter('output_file').value
        self.default_velocity = self.get_parameter('default_velocity').value
        self.waypoints = []

        self.sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        self.sub_point = self.create_subscription(
            PointStamped, '/clicked_point', self.point_callback, 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/waypoints_viz', 10)
        # Publish markers periodically so Foxglove always shows them
        self.timer = self.create_timer(0.5, self.publish_markers)

        self.get_logger().info('Waypoint clicker ready. Click points in Foxglove!')
        self.get_logger().info('Listening on /move_base_simple/goal AND /clicked_point')
        self.get_logger().info(f'Will save to: {self.output_file}')
        self.get_logger().info('Press Ctrl+C to save and exit.')

    def add_waypoint(self, x, y, yaw=0.0):
        self.waypoints.append([x, y, yaw, self.default_velocity])
        self.get_logger().info(
            f'[{len(self.waypoints)}] x={x:.4f}, y={y:.4f}, yaw={yaw:.4f}')
        self.publish_markers()

    def goal_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.add_waypoint(x, y, yaw)

    def point_callback(self, msg):
        self.add_waypoint(msg.point.x, msg.point.y)

    def publish_markers(self):
        ma = MarkerArray()
        for i, wp in enumerate(self.waypoints):
            # Sphere for each waypoint
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'waypoints'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = wp[0]
            m.pose.position.y = wp[1]
            m.pose.position.z = 0.1
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.3
            m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9)
            ma.markers.append(m)

            # Text label with index
            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = self.get_clock().now().to_msg()
            t.ns = 'waypoint_labels'
            t.id = i
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = wp[0]
            t.pose.position.y = wp[1]
            t.pose.position.z = 0.5
            t.scale.z = 0.3
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text = str(i + 1)
            ma.markers.append(t)

        # Line strip connecting waypoints
        if len(self.waypoints) >= 2:
            line = Marker()
            line.header.frame_id = 'map'
            line.header.stamp = self.get_clock().now().to_msg()
            line.ns = 'waypoint_path'
            line.id = 0
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.05
            line.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.8)
            for wp in self.waypoints:
                p = PointStamped().point
                p.x = wp[0]
                p.y = wp[1]
                p.z = 0.05
                line.points.append(p)
            ma.markers.append(line)

        self.marker_pub.publish(ma)

    def save_waypoints(self):
        if not self.waypoints:
            self.get_logger().warn('No waypoints to save!')
            return
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for wp in self.waypoints:
                writer.writerow(wp)
        self.get_logger().info(
            f'Saved {len(self.waypoints)} waypoints to {self.output_file}')


def main():
    rclpy.init()
    node = WaypointClicker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_waypoints()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

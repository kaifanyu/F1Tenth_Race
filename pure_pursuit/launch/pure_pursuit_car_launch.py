"""Launch pure pursuit node for real car with particle filter."""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('pure_pursuit')
    default_waypoint_file = os.path.join(pkg_dir, 'waypoints', 'waypoints_levine.csv')

    return LaunchDescription([
        Node(
            package='pure_pursuit',
            executable='pure_pursuit_node.py',
            name='pure_pursuit_node',
            output='screen',
            parameters=[{
                'waypoint_file': default_waypoint_file,
                'lookahead_distance': 1.0,
                'velocity': 1.0,
                'max_steering_angle': 0.4189,
                'wheelbase': 0.3302,
                'use_odom': False,  # car uses particle filter
                'speed_lookahead': False,
            }]
        ),
    ])

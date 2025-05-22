from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

import os


def generate_launch_description():

    config_node = os.path.join(
        get_package_share_directory('camera_cones_detector'),
        'config',
        'camera_cones_detector.yaml'
        )

    node=Node(
            package='camera_cones_detector',
            name='camera_cones_detector_node',
            executable='camera_cones_detector_node',
            parameters=[config_node, {'event_type': LaunchConfiguration('event_type')}]
        )

    return LaunchDescription(
        [           
            node
        ]
    )
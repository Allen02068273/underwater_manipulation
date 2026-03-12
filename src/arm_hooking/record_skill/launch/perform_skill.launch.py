from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression, TextSubstitution

from launch_ros.actions import Node

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # argument to choose between LTE (default) and KMP
    generator_arg = DeclareLaunchArgument(
        'learning_model',
        default_value='LTE',
        description='Choose learning model: LTE, DMP, or JA'
    )
    # argument to set serial port to connect to manipulator
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',
        description='Serial port to connect to manipulator (e.g. /dev/ttyUSB0)'
    )
    # argument to set AprilTag size in meters
    apriltag_size_arg = DeclareLaunchArgument(
        'apriltag_size',
        default_value='',
        description='AprilTag size in meters'
    )
    # argument to choose which frame to target
    target_frame_arg = DeclareLaunchArgument(
        'target_frame',
        default_value='target_2',
        description='Target on which to perform skill (e.g. target_2)'
    )
    # argument to set the time to complete the trajectory
    time_arg = DeclareLaunchArgument(
        "trajectory_duration",
        default_value='6.0',
        description="Time in seconds to complete the trajectory"
    )

    generator_executable = PythonExpression([
        '"', LaunchConfiguration('generator'), '_trajectory_generator"'
    ])

    # get path to the included launch file
    included_launch_path = os.path.join(
        get_package_share_directory('ra_core'),
        'reach_alpha_starter.launch.py'
    )

    # include the launch file with the arguments forwarded
    included_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(included_launch_path),
        launch_arguments=[
            ('serial_port', LaunchConfiguration('serial_port')),
            ('apriltag_size', LaunchConfiguration('apriltag_size')),
        ]
    )

    return LaunchDescription([
        generator_arg,
        serial_port_arg,
        apriltag_size_arg,
        target_frame_arg,
        time_arg,
        included_launch,
        Node(
            package='trajectory_generation',
            executable='lte_trajectory_generator',
            parameters=[{
                "trajectory_csv" : "data/trajectory.csv",
                "model" : LaunchConfiguration('learning_model'),
                "record_debug" : True,
                }]
            ),
        Node(
            package='record_skill',
            executable='perform_skill',
            parameters=[{
                "serial_port" : LaunchConfiguration('serial_port'),
                "target_frame" : LaunchConfiguration('target_frame'),
                "time" : LaunchConfiguration('trajectory_duration'),
                "record_debug" : True,
                }]
            )
            
    ])

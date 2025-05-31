from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

def generate_launch_description():
    # argument to set serial port to connect to manipulator
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyUSB0',
        description='Serial port to connect to manipulator (e.g. /dev/ttyUSB0)'
    )
    # argument to set AprilTag size in meters
    apriltag_size_arg = DeclareLaunchArgument(
        'apriltag_size',
        default_value='0.072',
        description='AprilTag size in meters'
    )

    return LaunchDescription([
        serial_port_arg,
        apriltag_size_arg,
        Node(
            package='bpl_passthrough',
            executable='serial_passthrough',
            parameters=[{"serial_port" : LaunchConfiguration('serial_port')}]
            ),
        Node(
            package='bpl_control',
            executable='end_effector_pose_publisher',
            parameters=[{
                "frame_id" : "alpha_base_link",
                "frequency" : 20
                }]
            ),
        Node(
            package='ra_core',
            executable='inverse_kinematics',
            parameters=[{"serial_port" : LaunchConfiguration('serial_port')}]
            ),
        Node(
            package='ra_core',
            executable='target_transforms',
            parameters=[{"apriltag_size" : LaunchConfiguration('apriltag_size')}]
            ),
        Node(
            package='arm_control',
            executable='arm_controller',
            parameters=[{"serial_port" : LaunchConfiguration('serial_port')}]
            )
            
    ])

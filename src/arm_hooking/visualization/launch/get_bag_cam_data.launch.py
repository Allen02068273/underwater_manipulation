from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
    bag_path_arg = DeclareLaunchArgument(
        "bag_path",
        default_value="data/bags/test_2025_8_16/rosbag2_2025_08_16-18_37_55",
        description="Path to the ROS2 bag to play"
    )

    bag_path = LaunchConfiguration("bag_path")

    return LaunchDescription([
        bag_path_arg,
        Node(
            package="visualization",
            executable="record_brov_camera_topic",
        ),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", bag_path, "--topics", "/my_camera/image_raw"],
        ),
    ])

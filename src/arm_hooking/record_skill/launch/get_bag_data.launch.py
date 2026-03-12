from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression, TextSubstitution
from launch.actions import ExecuteProcess

from launch_ros.actions import Node

def generate_launch_description():
    # argument to choose between LTE (default) and KMP
    generator_arg = DeclareLaunchArgument(
        'learning_model',
        default_value='LTE',
        description='Choose learning model: LTE, DMP, or JA'
    )
    # argument to set AprilTag size in meters
    apriltag_size_arg = DeclareLaunchArgument(
        'apriltag_size',
        default_value='0.112',  # 0.0715 in hook setup; 0.112 in valve setup
        description='AprilTag size in meters'
    )
    # argument to choose which frame to target
    target_frame_arg = DeclareLaunchArgument(
        'target_frame',
        default_value='valve',
        description='Target on which to perform skill (e.g. target_2 or valve)'
    )
    # argument to set the time to complete the trajectory
    time_arg = DeclareLaunchArgument(
        "trajectory_duration",
        default_value='6.0',
        description="Time in seconds to complete the trajectory"
    )
    # argument to set the file path to the bag to play
    bag_path_arg = DeclareLaunchArgument(
        "bag_path",
        default_value="data/bags/test_2025_8_16/rosbag2_2025_08_16-18_37_55",
        description="Path to the ROS2 bag to play"
    )

    # generator_executable = PythonExpression([
    #     '"', LaunchConfiguration('generator'), '_trajectory_generator"'
    # ])

    return LaunchDescription([
        generator_arg,
        apriltag_size_arg,
        target_frame_arg,
        time_arg,
        bag_path_arg,
        Node(  # sub: request_trajectory; pub: gen_trajectory
            package='trajectory_generation',
            executable='lte_trajectory_generator',
            parameters=[{
                "trajectory_csv" : "data/trajectory.csv",
                "model" : LaunchConfiguration('learning_model'),
                "record_debug" : True,
                }]
            ),
        Node(  # sub: gen_trajectory; pub: request_trajectory, command/km_command, command/ee_velocity, command/joint_positions
            package='record_skill',
            executable='perform_skill',
            parameters=[{
                "target_frame" : LaunchConfiguration('target_frame'),
                "time" : LaunchConfiguration('trajectory_duration'),
                "record_debug" : True,
                }]
            ),
        Node(  # sub: end_effector_pose, apriltag_pose; pub: none
            package='ra_core',
            executable='target_transforms',
            parameters=[{
                "apriltag_size" : LaunchConfiguration('apriltag_size'),
                }]
            ),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", LaunchConfiguration("bag_path"), "--topics", "/end_effector_pose", "/apriltag_pose"],
        ),
        # ExecuteProcess(  # use this one for bagged trajectory generator
        #     cmd=["ros2", "bag", "play", LaunchConfiguration("bag_path"), "--topics", "/end_effector_pose", "/apriltag_pose", "/gen_trajectory"],
        # ),
    ])

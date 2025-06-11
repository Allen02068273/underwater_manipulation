from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression, TextSubstitution

from launch_ros.actions import Node

def generate_launch_description():
    # Note: The real default argument values are set here in a PythonExpression
    #  so that this launch file can be called from other launch files
    #  with defaults set here if the other launch file passes an empty string ''.

    # argument to set serial port to connect to manipulator
    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='',
        description='Serial port to connect to manipulator (e.g. /dev/ttyUSB0)'
    )
    # argument to set AprilTag size in meters
    apriltag_size_arg = DeclareLaunchArgument(
        'apriltag_size',
        default_value='',
        description='AprilTag size in meters'
    )

    # create the actual parameters with conditional logic
    serial_port_param = PythonExpression([
        '"/dev/ttyUSB0" if "', LaunchConfiguration('serial_port'), '" == "" else "', LaunchConfiguration('serial_port'), '"'
    ])
    apriltag_size_param = PythonExpression([
        '0.0715 if "', LaunchConfiguration('apriltag_size'), '" == "" else float("', LaunchConfiguration('apriltag_size'), '")'
    ])

    return LaunchDescription([
        serial_port_arg,
        apriltag_size_arg,
        Node(
            package='ra_core',
            executable='ra_passthrough',
            parameters=[
                {"connection_type" : "serial"},
                {"serial_port" : serial_port_param},
                # {"ip_address" : "192.168.2.2"},
                # {"udp_port" : 6789},
                ]
            ),
        # Node(
        #     package='bpl_passthrough',
        #     executable='serial_passthrough',
        #     parameters=[{"serial_port" : serial_port_param}]
        #     ),
        # Node(
        #     package='bpl_passthrough',
        #     executable='udp_passthrough',
        #     parameters=[{"ip_address" : "192.168.2.2",
        #                  "port" : 6789}]
        #     ),
        Node(
            package='ra_core',
            executable='end_effector_pose_publisher',
            parameters=[{
                "frame_id" : "alpha_base_link",
                "frequency" : 20
                }]
            ),
        Node(
            package='ra_core',#'bpl_control',
            executable='control_node'
            ),
        Node(
            package='ra_core',
            executable='apriltag_pose_publisher',
            name='apriltag_publisher_reach_alpha',
            parameters=[{
                "apriltag_size" : apriltag_size_param,
                "camera_tf_frame" : "reach_alpha_camera",
                "display_feed" : True,
                }]
            ),
        Node(
            package='ra_core',
            executable='apriltag_pose_publisher',
            name='apriltag_publisher_brov',
            parameters=[{
                "apriltag_size" : apriltag_size_param,
                "camera_tf_frame" : "brov_camera",
                "display_feed" : True,
                }]
            ),
        Node(
            package='ra_core',
            executable='target_transforms',
            parameters=[{"apriltag_size" : apriltag_size_param}]
            ),
            
    ])

import rclpy
from rclpy.node import Node
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException, ConnectivityException
from geometry_msgs.msg import Pose
from tf2_geometry_msgs import do_transform_pose
import csv

class SkillRecorder(Node):

    def __init__(self):
        super().__init__('skill_recorder')

        self.localize_continuously = True  # True: more accurate but noisier; False: more precise but less accurate
        
        # transform listener for target/base/tool transforms
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.trans_base_to_target = None

        # set up CSV file
        csv_filename = "data/trajectory.csv"
        self.file = open(csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.csv_writer.writerow(['timestamp', 'x', 'y', 'z'])

        # debugging CSV file (gripper and AprilTag in robot frame)
        csv_filename_debug = "data/trajectory_debug.csv"
        self.file_debug = open(csv_filename_debug, mode='w', newline='')
        self.csv_writer_debug = csv.writer(self.file_debug)
        self.csv_writer_debug.writerow(['timestamp', 'tool_x', 'tool_y', 'tool_z', 'target_x', 'target_y', 'target_z'])

        # start timer for recording
        hz = 10  # entries added per second
        self.get_logger().info(f"Waiting for transform...")
        self.recorder_timer = self.create_timer(1/hz, self.recorder)

    def recorder(self):
        # get current timestamp
        timestamp = self.get_clock().now().to_msg().sec

        # try getting the transform of the tool in the target frame
        try:
            if self.trans_base_to_target is None or self.localize_continuously:
                self.trans_base_to_target = self._tf_buffer.lookup_transform('target_2', 'reach_alpha_base', rclpy.time.Time())
            tf_tool_in_base = self._tf_buffer.lookup_transform('reach_alpha_base', 'reach_alpha_tool', rclpy.time.Time())
        except (LookupException, ConnectivityException) as e:
            return
        
        p = Pose()
        p.position.x = tf_tool_in_base.transform.translation.x
        p.position.y = tf_tool_in_base.transform.translation.y
        p.position.z = tf_tool_in_base.transform.translation.z
        p = do_transform_pose(p, self.trans_base_to_target)

        # log pose to CSV
        self.csv_writer.writerow([
            timestamp,
            p.position.x,
            p.position.y,
            p.position.z
        ])

        # try getting robot frame transforms for debugging
        try:
            tf_tool_in_base = self._tf_buffer.lookup_transform('reach_alpha_base', 'reach_alpha_tool', rclpy.time.Time())
            tf_target_in_base = self._tf_buffer.lookup_transform('reach_alpha_base', 'target_2', rclpy.time.Time())
        except (LookupException, ConnectivityException) as e:
            self.get_logger().info(f"Waiting for transform {repr(e)}.")
            return

        # log tool and target positions in robot frame to debugging CSV
        self.csv_writer_debug.writerow([
            timestamp,
            tf_tool_in_base.transform.translation.x,
            tf_tool_in_base.transform.translation.y,
            tf_tool_in_base.transform.translation.z,
            tf_target_in_base.transform.translation.x,
            tf_target_in_base.transform.translation.y,
            tf_target_in_base.transform.translation.z
        ])

    def __del__(self):
        self.file.close()

def main(args=None):
    rclpy.init(args=args)
    node = SkillRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


import rclpy
from rclpy.node import Node
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException, ConnectivityException
from geometry_msgs.msg import Pose
from tf2_geometry_msgs import do_transform_pose
import csv
from pynput import keyboard

'''
Keyboard Controls
 l - localize
 o - start recording
 p - stop recording
'''

class SkillRecorder(Node):

    def __init__(self):
        super().__init__('skill_recorder')
        
        # transform listener for target/base/tool transforms
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.trans_base_to_target = None
        self.localize_continuously = True
        self.target_frame = 'target_2'

        # start listening to keyboard input
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()
        self.should_exit = False

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
        self.recording = False

    def on_key_press(self, key):
        try:
            if key.char == 'l':  # initialize localization
                try:
                    self.trans_base_to_target = self._tf_buffer.lookup_transform(self.target_frame, 'reach_alpha_base', rclpy.time.Time())
                    self.localize_continuously = False
                    self.get_logger().info(f"Successfully Localized")
                except (LookupException, ConnectivityException) as e:
                    self.get_logger().info(f"Localization Failed: {e}")
            elif key.char == 'o':  # start recording
                self.recording = True
                self.get_logger().info(f"Recording Started")
            elif key.char == 'p':  # stop recording and shut down node
                self.recording = False
                self.get_logger().info(f"Recording Stopped: shutting down node")
                self.listener.stop()
                self.file.close()
                self.file_debug.close()
                self.should_exit = True
        except AttributeError:
            pass  # handle special keys if necessary

    def recorder(self):
        if not self.recording:
            return
        
        self.get_logger().info(f"Recording")

        # get current timestamp
        timestamp = self.get_clock().now().to_msg().sec

        # try getting the necessary transforms
        try:
            if self.trans_base_to_target is None or self.localize_continuously:
                self.trans_base_to_target = self._tf_buffer.lookup_transform(self.target_frame, 'reach_alpha_base', rclpy.time.Time())
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

        self.get_logger().info(f"Data Written")

        # try getting robot frame transforms for debugging
        try:
            tf_tool_in_base = self._tf_buffer.lookup_transform('reach_alpha_base', 'reach_alpha_tool', rclpy.time.Time())
            tf_target_in_base = self._tf_buffer.lookup_transform('reach_alpha_base', self.target_frame, rclpy.time.Time())
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
        self.file_debug.close()

def main(args=None):
    rclpy.init(args=args)
    node = SkillRecorder()
    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


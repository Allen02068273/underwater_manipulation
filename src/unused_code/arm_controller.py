import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.publisher import Publisher
from pynput import keyboard

class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, 'command/km_command', 10)
        self.timer = self.create_timer(0.5, self.publish_pose)  # Publish every tenth of second

        # Initialize position (will be deleted in change to velocity)
        self.position_x = -0.00149
        self.position_y = -0.03118
        self.position_z = 0.10656

        # Start listening to keyboard input
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def on_key_press(self, key):
        # Handles keypress events to control the arm position
        self.get_logger().info(f"Detected keypress: {key.char}")
        try:
            if key.char == 'w':
                self.position_z += 0.01
            elif key.char == 's':
                self.position_z -= 0.01
            elif key.char == 'a':
                self.position_x -= 0.01
            elif key.char == 'd':
                self.position_x += 0.01
            elif key.char == 'q':
                self.position_y += 0.01
            elif key.char == 'e':
                self.position_y -= 0.01
        except AttributeError:
            pass  # Handle special keys if necessary

    def publish_pose(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "alpha_base_link"
        
        # Update pose based on current position
        pose.pose.position.x = self.position_x
        pose.pose.position.y = self.position_y
        pose.pose.position.z = self.position_z
        
        # Set the orientation as a quaternion (will be deleted in change to velocity)
        pose.pose.orientation.x = 0.00614
        pose.pose.orientation.y = 0.01772
        pose.pose.orientation.z = 0.69005
        pose.pose.orientation.w = 0.72352
        
        self.publisher_.publish(pose)
        self.get_logger().info(f"Publishing pose: {pose.pose.position.x}, {pose.pose.position.y}, {pose.pose.position.z}")

def main(args=None):
    rclpy.init(args=args)
    pose_publisher = PosePublisher()
    rclpy.spin(pose_publisher)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

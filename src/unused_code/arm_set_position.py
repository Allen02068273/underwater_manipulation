import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.publisher import Publisher

class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, 'command/km_command', 10)
        self.timer = self.create_timer(1.0, self.publish_pose)  # Publish every second

    def publish_pose(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "base_link"
        
        # Undocked, position (meters) and orientation
        pose.pose.position.x = 0.08401
        pose.pose.position.y = 0.00354
        pose.pose.position.z = 0.21660
        pose.pose.orientation.x = -0.86622
        pose.pose.orientation.y = -0.01336
        pose.pose.orientation.z = -0.01855
        pose.pose.orientation.w = -0.49914

        # Docked, position (meters) and orientation
        pose.pose.position.x = -0.03472
        pose.pose.position.y = -0.00113
        pose.pose.position.z = 0.10648
        pose.pose.orientation.x = -0.39911
        pose.pose.orientation.y = -0.01405
        pose.pose.orientation.z = -0.02397
        pose.pose.orientation.w = -0.91648
        
        self.publisher_.publish(pose)
        self.get_logger().info(f"Publishing pose: {pose.pose.position.x}, {pose.pose.position.y}, {pose.pose.position.z}")

def main(args=None):
    rclpy.init(args=args)
    pose_publisher = PosePublisher()
    rclpy.spin(pose_publisher)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

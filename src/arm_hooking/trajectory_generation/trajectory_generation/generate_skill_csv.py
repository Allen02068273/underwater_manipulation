import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException, ConnectivityException
import csv

class GenerateSkillCSV(Node):

    def __init__(self):
        super().__init__('generate_skill_csv')

        self.target_name = 'target_2'

        # transform listener for target/base/tool transforms
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # publisher for trajectory request
        self.request_publisher = self.create_publisher(Point, 'request_trajectory', 10)
        # subscriber to trajectory response
        self.subscription_trajectory = self.create_subscription(Float32MultiArray, 'gen_trajectory', self.trajectory_callback, 10)
        # timers for running scripts till completion
        self.request_timer = self.create_timer(1.0, self.send_initial_position)
        self.time_step = 0.1
        self.step_timer = None
        
        # trajectory and CSV storage file path
        self.trajectory = None
        self.csv_file_path = 'data/generated_trajectory.csv'

        # write CSV headers
        with open(self.csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y', 'z'])

    # request a trajectory based on initial position
    def send_initial_position(self):
        # try getting the transform of the tool in the target frame
        try:
            trans = self._tf_buffer.lookup_transform(self.target_name, 'reach_alpha_tool', rclpy.time.Time())
        except (LookupException, ConnectivityException) as e:
            self.get_logger().info(f"Waiting for transform {repr(e)}.")
            return
        # check for subscribers
        if self.request_publisher.get_subscription_count() == 0:
            self.get_logger().info("Waiting for subscriber to 'request_trajectory'.")
            return

        # convert to Point object and publish
        self.position = Point()
        self.position.x = float(trans.transform.translation.x)
        self.position.y = float(trans.transform.translation.y)
        self.position.z = float(trans.transform.translation.z)
        self.request_publisher.publish(self.position)
        self.request_timer.cancel()
        self.get_logger().info(f"Requested trajectory with initial position {self.position}.")

    def trajectory_callback(self, msg):
        # ensure only one trajectory is processed
        if self.trajectory is not None:
            return
        
        self.get_logger().info("Received trajectory from generator.")
        
        # received data length must be a multiple of 3 for x, y, z
        if len(msg.data) % 3 != 0:
            self.get_logger().error("Trajectory data length is not a multiple of 3.")
            return
        
        # convert the flattened data into a list of Point messages
        self.trajectory = []
        for i in range(0, len(msg.data), 3):
            pt = Point()
            pt.x = msg.data[i]
            pt.y = msg.data[i+1]
            pt.z = msg.data[i+2]
            self.trajectory.append(pt)

        self.write_trajectory_to_csv()
        self.destroy_node()

    def write_trajectory_to_csv(self):
        with open(self.csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for pt in self.trajectory:
                writer.writerow([pt.x, pt.y, pt.z])
        self.get_logger().info(f"Trajectory data written to {self.csv_file_path}")

def main(args=None):
    rclpy.init(args=args)
    node = GenerateSkillCSV()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

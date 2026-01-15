import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import csv
from datetime import datetime

class CurrentRecorder(Node):

    def __init__(self):
        super().__init__('current_recorder')

        # output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.declare_parameter('output_path', f'data/current_recordings/current_{timestamp}.csv')
        output_path = self.get_parameter('output_path').value

        # set up CSV file
        self.file = open(output_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.csv_writer.writerow(['Gripper', 'Joint B', 'Joint C', 'Joint D', 'Base Joint'])

        self.subscription = self.create_subscription(Float32MultiArray, 'current', self.record_current, 10)

        self.get_logger().info(f"Recording RA5 joint currents to {output_path}")

    def record_current(self, msg):
        currents = msg.data
        self.csv_writer.writerow(currents)

    def __del__(self):
        self.file.close()


def main(args=None):
    rclpy.init(args=args)
    node = CurrentRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

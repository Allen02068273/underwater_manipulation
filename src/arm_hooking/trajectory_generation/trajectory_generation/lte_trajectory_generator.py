import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from scipy import interpolate
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from trajectory_generation.utils import LTE, JA, DMP

import csv

class LTETrajectoryNode(Node):
    def __init__(self):
        super().__init__('lte_trajectory_generator')
        
        # parameters
        self.declare_parameter('trajectory_csv', 'data/trajectory.csv')
        self.declare_parameter('model', 'LTE')  # 'LTE', 'DMP', or 'JA'
        self.declare_parameter('record_debug', True)
        csv_file = self.get_parameter('trajectory_csv').value
        self.model = self.get_parameter('model').value
        self.record_debug = self.get_parameter('record_debug').value

        # debugging CSV files
        self.csv_files = {}
        self.csv_writers = {}

        if self.record_debug:
            csv_filenames = ["raw", "smoothed", "generalized"]
            for name in csv_filenames:
                file_path = f"data/debug_trajectories/{name}.csv"
                f = open(file_path, mode='w', newline='')
                self.csv_files[name] = f
                writer = csv.writer(f)
                self.csv_writers[name] = writer
                writer.writerow(['timestamp', 'x', 'y', 'z'])
        self.logged_generalized = False
        
        # load trajectory from CSV
        self.traj = self.load_trajectory_from_csv(csv_file)
        self.write_traj_to_csv('raw', self.traj)

        # smooth trajectory
        sample_count = 20
        self.traj = self.smooth_trajectory(self.traj, sample_count)
        self.write_traj_to_csv('smoothed', self.traj)
        
        # subscriber to XYZ initial positions
        self.subscription = self.create_subscription(
            Point,
            'request_trajectory',
            self.request_callback,
            10
        )

        # publisher for trajectory response
        self.publisher = self.create_publisher(
            Float32MultiArray,
            'gen_trajectory',
            10
        )
        
        self.get_logger().info("LTE Trajectory Node Initialized.")

    def load_trajectory_from_csv(self, csv_file):
        # Load trajectory from CSV file and verify required columns
        try:
            df = pd.read_csv(csv_file)
            required_columns = {'x', 'y', 'z'}
            if not required_columns.issubset(df.columns):
                self.get_logger().error(f"CSV file missing required columns: {required_columns - set(df.columns)}. Shutting down.")
                rclpy.shutdown()
                return None
            
            traj = df[['x', 'y', 'z']].to_numpy()
            self.get_logger().info(f"Loaded trajectory from {csv_file}")
            return traj
        except Exception as e:
            self.get_logger().error(f"Failed to load CSV: {e}. Shutting down.")
            rclpy.shutdown()
            return None

    def smooth_trajectory(self, traj, sample_count):
        # Ensure trajectory has at least 2 points
        if traj.shape[0] < 2:
            self.get_logger().info("Trajectory must have at least 2 points for smoothing.")
            return traj

        # Extract x, y, z components
        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]

        px = interpolate.interp1d(np.linspace(0, 1, len(x)), x)
        x_smooth = px(np.linspace(0, 1, sample_count))
        py = interpolate.interp1d(np.linspace(0, 1, len(y)), y)
        y_smooth = py(np.linspace(0, 1, sample_count))
        pz = interpolate.interp1d(np.linspace(0, 1, len(z)), z)
        z_smooth = pz(np.linspace(0, 1, sample_count))

        # Combine the smoothed components
        smoothed_traj = np.vstack((x_smooth, y_smooth, z_smooth)).T

        return smoothed_traj

    def request_callback(self, msg):
        # generate LTE trajectory starting from input point
        start_point = np.array([msg.x, msg.y, msg.z])
        end_point = np.array([self.traj[-1, 0], self.traj[-1, 1], self.traj[-1, 2]])

        if self.model=="LTE":
            # this constraint point should be directly over the hook; without it, the arm may not reach far enough on reproductions
            mid_index = int(.65 * len(self.traj))
            mid_point = np.array([self.traj[mid_index, 0], self.traj[mid_index, 1], self.traj[mid_index, 2]])
            new_traj = LTE(self.traj, [start_point, mid_point, end_point], [0, mid_index, len(self.traj) - 1])
        elif self.model=="DMP":
            new_traj = DMP(self.traj, [start_point, end_point], [0, len(self.traj) - 1])
        elif self.model=="JA":
            new_traj = JA(self.traj, [start_point, end_point], [0, len(self.traj) - 1], lmbda=75)

        # publish the generated trajectory
        traj_msg = Float32MultiArray()
        traj_msg.data = new_traj.flatten().tolist()
        self.publisher.publish(traj_msg)
        if not self.logged_generalized:
            self.write_traj_to_csv('generalized', new_traj)
            self.logged_generalized = True
        self.get_logger().info("Published LTE trajectory.")

    def write_traj_to_csv(self, csv_name, traj):
        if self.record_debug:
            for x in traj:
                self.csv_writers[csv_name].writerow([0, x[0], x[1], x[2]])

    def __del__(self):
        for file in self.csv_files.values():
            file.close()

def main(args=None):
    rclpy.init(args=args)
    node = LTETrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
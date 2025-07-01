import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from scipy import interpolate
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from trajectory_generation.utils.lte import LTE

import csv

class LTETrajectoryNode(Node):
    def __init__(self):
        super().__init__('lte_trajectory_generator')
        
        # parameters
        self.declare_parameter('trajectory_csv', 'data/trajectory.csv')
        self.declare_parameter('record_debug', True)
        csv_file = self.get_parameter('trajectory_csv').value
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
        
        # load trajectory from CSV and smooth
        self.traj = self.load_trajectory_from_csv(csv_file)
        self.try_write_traj_to_csv('raw', self.traj)
        sample_count = 20
        self.traj = self.smooth_trajectory(self.traj, sample_count)
        self.try_write_traj_to_csv('smoothed', self.traj)
        
        self.fixed_weight = 1e9
        
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

        # # Create a parameter t to represent the progression along the trajectory
        # t = np.linspace(0, 1, len(traj))

        # # Fit a B-spline to each component
        # tck_x = interpolate.splrep(t, x, s=0)
        # tck_y = interpolate.splrep(t, y, s=0)
        # tck_z = interpolate.splrep(t, z, s=0)

        # # Resample the trajectory with the new number of points
        # t_new = np.linspace(0, 1, sample_count)
        # x_smooth = interpolate.splev(t_new, tck_x)
        # y_smooth = interpolate.splev(t_new, tck_y)
        # z_smooth = interpolate.splev(t_new, tck_z)

        # Compute distances between consecutive points
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        arc_length = np.concatenate(([0], np.cumsum(distances)))

        from scipy.interpolate import CubicSpline

        # Fit splines
        spline_x = CubicSpline(arc_length, x)
        spline_y = CubicSpline(arc_length, y)
        spline_z = CubicSpline(arc_length, z)

        # Generate equally spaced arc lengths
        s_new = np.linspace(0, arc_length[-1], sample_count)
        x_smooth = spline_x(s_new)
        y_smooth = spline_y(s_new)
        z_smooth = spline_z(s_new)

        # Combine the smoothed components
        smoothed_traj = np.vstack((x_smooth, y_smooth, z_smooth)).T

        return smoothed_traj

    def request_callback(self, msg):
        # generate LTE trajectory starting from input point
        start_point = np.array([msg.x, msg.y, msg.z])
        end_point = np.array([self.traj[-1, 0], self.traj[-1, 1], self.traj[-1, 2]])
        new_traj = LTE(self.traj, [start_point, end_point], [0, len(self.traj) - 1])

        # publish the generated trajectory
        traj_msg = Float32MultiArray()
        traj_msg.data = new_traj.flatten().tolist()
        self.publisher.publish(traj_msg)
        self.try_write_traj_to_csv('generalized', new_traj)
        self.get_logger().info("Published LTE trajectory.")

    def try_write_traj_to_csv(self, csv_name, traj):
        if self.record_debug:
            for x in traj:
                self.csv_writers[csv_name].writerow([0, x[0], x[1], x[2]])

    def __del__(self):
        for file in self.csv_files:
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
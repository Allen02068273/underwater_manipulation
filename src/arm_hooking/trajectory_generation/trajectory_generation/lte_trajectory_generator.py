import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from scipy import interpolate
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray

class LTETrajectoryNode(Node):
    def __init__(self):
        super().__init__('lte_trajectory_generator')
        
        # Declare and get CSV parameter
        self.declare_parameter('trajectory_csv', 'data/trajectory.csv')
        csv_file = self.get_parameter('trajectory_csv').value
        
        # Load trajectory from CSV and smooth
        self.traj = self.load_trajectory_from_csv(csv_file)
        sample_count = 20
        self.traj = self.smooth_trajectory(self.traj, sample_count)
        
        self.fixed_weight = 1e9
        
        # Subscriber to XYZ initial positions
        self.subscription = self.create_subscription(
            Point,
            'request_trajectory',
            self.request_callback,
            10
        )

        # Publisher for trajectory response
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

        # Create a parameter t to represent the progression along the trajectory
        t = np.linspace(0, 1, len(traj))

        # Fit a B-spline to each component
        tck_x = interpolate.splrep(t, x, s=0)
        tck_y = interpolate.splrep(t, y, s=0)
        tck_z = interpolate.splrep(t, z, s=0)

        # Resample the trajectory with the new number of points
        t_new = np.linspace(0, 1, sample_count)
        x_smooth = interpolate.splev(t_new, tck_x)
        y_smooth = interpolate.splev(t_new, tck_y)
        z_smooth = interpolate.splev(t_new, tck_z)

        # Combine the smoothed components
        smoothed_traj = np.vstack((x_smooth, y_smooth, z_smooth)).T

        return smoothed_traj


    def request_callback(self, msg):
        # generate LTE trajectory starting from input point
        start_point = np.array([msg.x, msg.y, msg.z])
        end_point = np.array([self.traj[-1, 0], self.traj[-1, 1], self.traj[-1, 2]])
        new_traj = self.LTE(self.traj, [start_point, end_point], [0, len(self.traj) - 1])

        # publish the generated trajectory
        traj_msg = Float32MultiArray()
        traj_msg.data = new_traj.flatten().tolist()
        self.publisher.publish(traj_msg)
        self.get_logger().info("Published LTE trajectory.")

    def LTE(self, X, C, inds):
        n_pts, n_dims = np.shape(X)
        L = self.generate_laplacian(n_pts)
        D = self.laplacian_transform(L, X)
        P = self.append_constraints(D, C, inds)
        new_L = self.append_weights(L, C, inds)
        new_X = self.cartesian_transform(new_L, P)
        return new_X

    def generate_laplacian(self, n_pts):
        L = 2.*np.diag(np.ones((n_pts,))) - np.diag(np.ones((n_pts-1,)),1) - np.diag(np.ones((n_pts-1,)),-1)
        L[0,1] = -2.
        L[-1,-2] = -2.
        L = L / 2.
        return L

    def laplacian_transform(self, L, X):
        D = np.matmul(L, X)
        return D
        
    def append_constraints(self, D, C, inds):
        for const in C:
            D = np.vstack((D, const.reshape((1, len(const))) * self.fixed_weight))
        return D

    def append_weights(self, L, C, inds):
        n_pts = len(L)
        for i in inds:
            to_append = np.ones((1, n_pts))
            to_append[0, i] = self.fixed_weight
            L = np.vstack((L, to_append))
        return L
        
    def cartesian_transform(self, new_L, P):
        new_traj, _, _, _ = np.linalg.lstsq(new_L, P, rcond=-1)
        return new_traj

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
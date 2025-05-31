import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from scipy import interpolate
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from movement_primitives.promp import ProMP
import h5py
import os

class PotentialField():
    def __init__(self):
        self.obstacles = []

    def add_obstacle(self, position, strength):
        """
        Adds a point obstacle to the potential field.

        Args:
            position (np.ndarray): Position of the obstacle in space.
            strength (float): How strongly the obstacle repels.
        """
        self.obstacles.append((position, strength))
    
    def get_field_vector(self, position):
        """
        Computes the repulsive field vector at a given position.

        Args:
            position (np.ndarray): The point at which to compute the field vector.

        Returns:
            np.ndarray: The resulting field vector due to all obstacles.
        """
        field_vector = np.zeros_like(position)
        for obs_position, obs_strength in self.obstacles:
            diff = position - obs_position
            dist = np.linalg.norm(diff) + 1e-8
            field_vector += obs_strength * diff / (dist * (dist + 0.01))
        return field_vector

class ProMPTrajectoryNode(Node):
    def __init__(self):
        super().__init__('promp_trajectory_generator')
        
        # Declare and get CSV parameters
        # self.declare_parameter('trajectory_csvs', ['data/skill_data1.csv', 'data/skill_data2.csv'])
        # csv_files = self.get_parameter('trajectory_csvs').value

        # Load and smooth trajectories from multiple CSV files
        csv_files = ['data/demos/trajectory_1.csv', 'data/demos/trajectory_2.csv', 'data/demos/trajectory_3.csv', 'data/demos/trajectory_4.csv']
        self.sample_count = 100
        trajectories = self.load_and_smooth_trajectories(csv_files, self.sample_count)

        # Data for graphing
        self.trajectories = trajectories
        self.graph_displayed = False

        # Initialize ProMP model
        self.promp_model = self.initialize_promp_with_multiple_demos(trajectories)

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

        self.obstacle_field = PotentialField()
        self.obstacle_field.add_obstacle(np.array([-0.10, 0, -0.01]), 0.0008)
        
        self.get_logger().info("ProMP Trajectory Node Initialized.")

    def load_and_smooth_trajectories(self, csv_files, sample_count):
        trajectories = []
        for csv_file in csv_files:
            traj = self.load_trajectory_from_csv(csv_file)
            if traj is not None:
                smoothed_traj = self.smooth_trajectory(traj, sample_count)
                trajectories.append(smoothed_traj)
        
        # Convert to a NumPy array with shape (n_demos, n_steps, n_dims)
        return np.array(trajectories)
    
    def initialize_promp_with_multiple_demos(self, trajectories):
        if len(trajectories) == 0:
            self.get_logger().error("No valid trajectories provided. Shutting down.")
            rclpy.shutdown()
            return None

        n_demos, n_steps, n_dims = trajectories.shape

        # Create a time array normalized between 0 and 1
        T = np.linspace(0, 1, n_steps)

        # Initialize ProMP with correct dimensions
        promp = ProMP(n_dims=n_dims, n_weights_per_dim=10)

        # Train ProMP with multiple demonstrations
        promp.imitate([T] * n_demos, trajectories)

        return promp

    def load_trajectory_from_csv(self, csv_file):
        # Load trajectory from CSV file and verify required columns
        try:
            df = pd.read_csv(csv_file)
            required_columns = {'x', 'y', 'z'}
            if not required_columns.issubset(df.columns):
                self.get_logger().error(f"CSV file {csv_file} missing required columns: {required_columns - set(df.columns)}.")
                return None
            
            traj = df[['x', 'y', 'z']].to_numpy()
            self.get_logger().info(f"Loaded trajectory from {csv_file}")
            return traj
        except Exception as e:
            self.get_logger().error(f"Failed to load CSV {csv_file}: {e}")
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

        self.get_logger().info(f"Smoothed data with shape: {smoothed_traj.shape}")

        # Save smoothed trajectory to HDF5 file
        # h5_filename = f"data/{os.path.basename(csv_file).replace('.csv', '_smooth.h5')}"
        # try:
        #     with h5py.File(h5_filename, "w") as file:
        #         dataset = file.create_dataset(
        #             "smoothed_trajectory",
        #             data=smoothed_traj,
        #             dtype=np.float64
        #         )
        #     self.get_logger().info(f"Smoothed trajectory saved to {h5_filename}")
        # except Exception as e:
        #     self.get_logger().error(f"Failed to save smoothed trajectory: {e}")
        return smoothed_traj
    
    def adjust_trajectory_for_obstacles(self, traj):
        adjusted_traj = []
        threshold = 0.012

        for point in traj:
            field_vec = self.obstacle_field.get_field_vector(point)
            if np.linalg.norm(field_vec) > threshold:
                adjusted_point = point + 200*(field_vec * (np.linalg.norm(field_vec) - threshold))
            else:
                adjusted_point = point
            adjusted_traj.append(adjusted_point)

        adjusted_traj = np.array(adjusted_traj)
        return adjusted_traj

    def initialize_promp(self, trajectories):
        promp = ProMP(n_dims=trajectories[0].shape[1], n_weights_per_dim=50)
        for traj in trajectories:
            T = np.linspace(0, 1, len(traj))
            promp.add_demonstration(T, traj)
        promp.estimate_distribution()
        return promp

    def request_callback(self, msg):
        # Define start and end points
        start_point = np.array([msg.x, msg.y, msg.z])

        # Time steps for ProMP (normalized from 0 to 1)
        T = np.linspace(0, 1, self.sample_count)

        # Apply ProMP conditioning on the start point
        conditioned_promp = self.promp_model.condition_position(start_point, t=0)

        # Generate a new trajectory with the updated constraints
        new_traj = conditioned_promp.mean_trajectory(T)

        # Apply a potential field to push trajectory away from obstacles
        adjusted_traj = self.adjust_trajectory_for_obstacles(new_traj)

        # Publish the new trajectory
        traj_msg = Float32MultiArray()
        traj_msg.data = adjusted_traj.flatten().tolist()
        self.publisher.publish(traj_msg)
        self.get_logger().info("Published ProMP trajectory with start constraint.")
        self.traj_to_graph(new_traj, adjusted_traj)
    
    def traj_to_graph(self, new_traj, adjusted_traj):
        if self.graph_displayed:
            return
        self.graph_displayed = True
        
        import matplotlib.pyplot as plt

        # Plot setup
        fig, ax = plt.subplots(figsize=(10, 6))

        # 1. Plot each trajectory in the list
        for idx, traj in enumerate(self.trajectories):
            ax.plot(traj[:, 0], traj[:, 2], label=f'Trajectory {idx+1}', alpha=0.5)

        # 2. Overlay the potential field (XZ-plane)
        # Create a grid for field visualization
        x_range = np.linspace(-0.2, 0, 20)
        z_range = np.linspace(-0.05, 0.1, 20)
        X, Z = np.meshgrid(x_range, z_range)
        U = np.zeros_like(X)
        W = np.zeros_like(Z)

        scale = 0.01
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pos = np.array([X[i, j], 0.0, Z[i, j]])  # y=0 slice of the field
                vec = scale * self.obstacle_field.get_field_vector(pos)
                U[i, j] = vec[0]  # x component
                W[i, j] = vec[2]  # z component

        ax.quiver(X, Z, U, W, color='gray', alpha=0.6, label='Potential Field')

        # 3. Plot the new and adjusted trajectories
        ax.plot(new_traj[:, 0], new_traj[:, 2], 'b--', linewidth=2, label='New Trajectory')
        ax.plot(adjusted_traj[:, 0], adjusted_traj[:, 2], 'r-', linewidth=2, label='Adjusted Trajectory')

        # Final plot settings
        ax.set_title("Trajectories with Repulsive Potential Field (XZ plane)")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.axis("equal")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = ProMPTrajectoryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
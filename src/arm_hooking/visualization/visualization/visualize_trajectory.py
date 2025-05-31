import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('trajectory_visualizer')

        # transform listener for target/base/tool transforms
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # publisher for trajectory request
        self.request_publisher = self.create_publisher(Point, 'request_trajectory', 10)
        # subscriber to trajectory response
        self.subscription_trajectory = self.create_subscription(Float32MultiArray, 'gen_trajectory', self.trajectory_callback, 20)

        # store received trajectory
        self.generated_trajectory = None
        self.original_trajectories = self.load_original_trajectories()

        # data for plotting trajectories
        self.fig = None
        self.axes = None
        self.colors = None

        # start timer for requesting new trajectories
        hz = 5  # updates per second
        self.recorder_timer = self.create_timer(1/hz, self.send_initial_position)

    def load_original_trajectories(self):
        """Loads multiple original trajectories from CSV files."""
        trajectories = []
        # Get the list of CSV files, each containing one trajectory
        csv_files = ['data/trajectory.csv']
        csv_files = ['data/demos/trajectory_1.csv', 'data/demos/trajectory_2.csv', 'data/demos/trajectory_3.csv', 'data/demos/trajectory_4.csv']

        self.target_name = 'target_2'

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                required_columns = {'x', 'y', 'z'}
                if not required_columns.issubset(df.columns):
                    self.get_logger().error(f"CSV file {csv_file} missing required columns: {required_columns - set(df.columns)}")
                    continue
                traj = df[['x', 'y', 'z']].to_numpy()
                self.get_logger().info(f"Loaded original trajectory from {csv_file}")
                trajectories.append(traj)
            except Exception as e:
                self.get_logger().error(f"Failed to load original trajectory from {csv_file}: {e}")

        if len(trajectories) == 0:
            self.get_logger().error("No original trajectories were loaded.")
            return None
        return trajectories

    # request a trajectory based on the current position
    def send_initial_position(self):
        # try getting the transform of the tool in the target frame
        try:
            trans = self._tf_buffer.lookup_transform(self.target_name, 'reach_alpha_tool', rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
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
        self.get_logger().info(f"Requested trajectory with initial position {self.position}.")
        if self.fig is None:
            self.initialize_plot()
            self.get_logger().info("Plots initialized.")

    def trajectory_callback(self, msg):
        """Receives the trajectory and plots both original and generated trajectories."""
        self.get_logger().info("Received trajectory from generator node.")

        if len(msg.data) % 3 != 0:
            self.get_logger().error("Trajectory data length is not a multiple of 3.")
            return

        # Convert to NumPy array
        self.generated_trajectory = np.array(msg.data).reshape(-1, 3)
        self.plot_trajectories()

    def initialize_plot(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.colors = ['red', 'orange', 'green', 'purple', 'brown']

    def plot_trajectories(self):
        # Clear previous plots
        for ax in self.axes:
            ax.clear()

        # Extract generated trajectory coordinates
        x_gen, y_gen, z_gen = self.generated_trajectory[:, 0], self.generated_trajectory[:, 1], self.generated_trajectory[:, 2]

        # Compute axis limits across all trajectories
        x_min, x_max = x_gen.min(), x_gen.max()
        y_min, y_max = -y_gen.min(), -y_gen.max()
        z_min, z_max = z_gen.min(), z_gen.max()

        for orig_traj in self.original_trajectories:
            x_orig, y_orig, z_orig = orig_traj[:, 0], orig_traj[:, 1], orig_traj[:, 2]
            x_min, x_max = min(x_min, x_orig.min()), max(x_max, x_orig.max())
            y_min, y_max = min(y_min, -y_orig.min()), max(y_max, -y_orig.max())
            z_min, z_max = min(z_min, z_orig.min()), max(z_max, z_orig.max())

        all_min = min(x_min, y_min, z_min, 0)
        all_max = max(x_max, y_max, z_max, 0)
        padding = 0.05 * (all_max - all_min)
        all_min, all_max = all_min - padding, all_max + padding

        def plot_view(ax, x_data_list, y_data_list, x_data2, y_data2, xlabel, ylabel, title):
            """Plots multiple original trajectories and the generated trajectory."""
            for i, (x_data1, y_data1) in enumerate(zip(x_data_list, y_data_list)):
                color = self.colors[i % len(self.colors)]
                ax.plot(x_data1, y_data1, label=f'Original Demo {i+1}', color=color, marker='o', markersize=3, linestyle='-')

            ax.plot(x_data2, y_data2, label='Generated Trajectory', color='blue', marker='o', markersize=3, linestyle='-')
            ax.scatter(0, 0, color='black', marker='x', s=100, label="AprilTag")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlim(all_min, all_max)
            ax.set_ylim(all_min, all_max)
            ax.legend()
            ax.grid()

        x_orig_list = [orig_traj[:, 0] for orig_traj in self.original_trajectories]
        y_orig_list = [orig_traj[:, 1] for orig_traj in self.original_trajectories]
        z_orig_list = [orig_traj[:, 2] for orig_traj in self.original_trajectories]

        plot_view(self.axes[0], [-y for y in y_orig_list], z_orig_list, -y_gen, z_gen, "Y (Horizontal)", "Z (Height)", "Back View")
        plot_view(self.axes[1], x_orig_list, z_orig_list, x_gen, z_gen, "X (Depth)", "Z (Height)", "Side View")
        plot_view(self.axes[2], [-y for y in y_orig_list], x_orig_list, -y_gen, x_gen, "Y (Horizontal)", "X (Depth)", "Top View")

        plt.draw()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

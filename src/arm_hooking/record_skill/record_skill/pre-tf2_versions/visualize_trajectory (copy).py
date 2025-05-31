import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
import time
import cv2
import apriltag
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('trajectory_visualizer')

        # Publisher for trajectory request
        self.request_publisher = self.create_publisher(Point, 'request_trajectory', 10)
        # Subscriber to trajectory response
        self.subscription_trajectory = self.create_subscription(Float32MultiArray, 'gen_trajectory', self.trajectory_callback, 20)

        # GStreamer pipeline for RTSP camera stream
        rtsp_url = "rtsp://admin:@192.168.2.10:554/stream=1"
        self.gst_pipeline = f"rtspsrc location={rtsp_url} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error("Error: Couldn't open the RTSP stream.")
            rclpy.shutdown()
            return

        # Load camera calibration data
        with np.load("data/camera_calibration.npz") as data:
            self.intrinsic_matrix = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]

        # Initialize the AprilTag detector
        self.detector = apriltag.Detector()
        tag_size = 0.1651  # Tag size in meters
        self.tag_3d_points = np.array([[-tag_size/2, -tag_size, 0],
                                       [ tag_size/2, -tag_size, 0],
                                       [ tag_size/2, 0, 0],
                                       [-tag_size/2, 0, 0]], dtype=np.float32)

        # Store received trajectory
        self.generated_trajectory = None
        self.original_trajectories = self.load_original_trajectories()

        # Wait for subscribers to connect
        timeout = 5  # seconds
        start_time = time.time()
        while self.request_publisher.get_subscription_count() == 0:
            if time.time() - start_time > timeout:
                self.get_logger().warn("No subscribers found for 'request_trajectory', sending request anyway.")
                break
            self.get_logger().info("Waiting for subscriber...")
            time.sleep(0.5)

        # Request trajectory based on camera input
        self.send_initial_position()

    def load_original_trajectories(self):
        """Loads multiple original trajectories from CSV files."""
        trajectories = []
        # Get the list of CSV files from a parameter (you could also hard-code this list)
        csv_files = ['data/demos/skill_data1.csv', 'data/demos/skill_data2.csv', 'data/demos/skill_data3.csv']

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                required_columns = {'gripper_x', 'gripper_y', 'gripper_z'}
                if not required_columns.issubset(df.columns):
                    self.get_logger().error(f"CSV file {csv_file} missing required columns: {required_columns - set(df.columns)}")
                    continue
                traj = df[['gripper_x', 'gripper_y', 'gripper_z']].to_numpy()
                self.get_logger().info(f"Loaded original trajectory from {csv_file}")
                trajectories.append(traj)
            except Exception as e:
                self.get_logger().error(f"Failed to load original trajectory from {csv_file}: {e}")

        if len(trajectories) == 0:
            self.get_logger().error("No original trajectories were loaded.")
            return None
        return trajectories

    def send_initial_position(self):
        """Detects AprilTags in the camera stream and requests a trajectory."""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Error: Couldn't read frame from stream.")
            rclpy.shutdown()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)

        for tag in tags:
            corners = tag.corners

            # Estimate pose
            image_points = np.array(corners, dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(self.tag_3d_points, image_points, self.intrinsic_matrix, self.dist_coeffs)
            if not success:
                continue

            # Convert to robot frame and account for mounted camera offset and hook offset
            tvec = np.array([tvec[2] - (0.026 + 0.120), -tvec[0], -tvec[1] + 0.033])
            R, _ = cv2.Rodrigues(rvec)
            camera_position_in_tag = -np.transpose(R) @ tvec

            # Convert to Point message and publish
            pos = camera_position_in_tag.flatten()
            initial_position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            self.request_publisher.publish(initial_position)
            self.get_logger().info(f"Requested trajectory with initial position {initial_position}.")

            return

    def trajectory_callback(self, msg):
        """Receives the trajectory and plots both original and generated trajectories."""
        self.get_logger().info("Received trajectory from generator node.")

        if len(msg.data) % 3 != 0:
            self.get_logger().error("Trajectory data length is not a multiple of 3.")
            return

        # Convert to NumPy array
        self.generated_trajectory = np.array(msg.data).reshape(-1, 3)
        self.plot_trajectories()

    def plot_trajectories(self):
        """Plots multiple original demonstrations alongside the generated trajectory."""
        if self.original_trajectories is None or len(self.original_trajectories) == 0:
            self.get_logger().error("No original trajectories available.")
            return

        if self.generated_trajectory is None:
            self.get_logger().error("No generated trajectory available.")
            return

        fig = plt.figure(figsize=(18, 6))

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

        colors = ['red', 'orange', 'green', 'purple', 'brown']  # Color options for multiple demos

        def plot_view(ax, x_data_list, y_data_list, x_data2, y_data2, xlabel, ylabel, title):
            """Plots multiple original trajectories and the generated trajectory."""
            for i, (x_data1, y_data1) in enumerate(zip(x_data_list, y_data_list)):
                color = colors[i % len(colors)]  # Cycle through colors
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

        ax1 = fig.add_subplot(1, 3, 1)
        plot_view(ax1, [-y for y in y_orig_list], z_orig_list, -y_gen, z_gen, "Y (Horizontal)", "Z (Height)", "Back View")

        ax2 = fig.add_subplot(1, 3, 2)
        plot_view(ax2, x_orig_list, z_orig_list, x_gen, z_gen, "X (Depth)", "Z (Height)", "Side View")

        ax3 = fig.add_subplot(1, 3, 3)
        plot_view(ax3, [-y for y in y_orig_list], x_orig_list, -y_gen, x_gen, "Y (Horizontal)", "X (Depth)", "Top View")

        plt.tight_layout()
        plt.show()

    def __del__(self):
        self.cap.release()


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

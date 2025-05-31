import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray

import cv2
import apriltag
import numpy as np
from scipy.spatial.transform import Rotation as R

from bplprotocol import BPLProtocol, PacketID
import serial

from threading import Thread, Lock

class SkillPerformer(Node):

    def __init__(self):
        super().__init__('skill_performer')

        # parameter for serial port name to connect to the manipulator
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        serial_port_name = self.get_parameter('serial_port').value

        # publisher for trajectory request
        self.request_publisher = self.create_publisher(Point, 'request_trajectory', 10)
        # subscriber to trajectory response
        self.subscription_trajectory = self.create_subscription(Float32MultiArray, 'gen_trajectory', self.trajectory_callback, 10)
        # subscriber to end effector pose
        self.subscription_pose = self.create_subscription(PoseStamped, 'end_effector_pose', self.pose_callback, 10)
        # publisher for end effector pose
        self.pose_publisher = self.create_publisher(PoseStamped, 'command/km_command', 10)
        # timers for running scripts till completion
        self.request_timer = self.create_timer(1.0, self.send_initial_position)
        self.time_step = 0.1
        self.step_timer = None

        # GStreamer pipeline for low-latency RTSP stream
        rtsp_url = "rtsp://admin:@192.168.2.10:554/stream=1"
        self.gst_pipeline = f"rtspsrc location={rtsp_url} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
        # initialize OpenCV with the GStreamer pipeline
        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
        # check if the video capture was successfully opened
        if not self.cap.isOpened():
            self.get_logger().error("Error: Couldn't open the RTSP stream.")
            rclpy.shutdown()
            return None

        # Load camera calibration data
        with np.load("data/camera_calibration.npz") as data:
            self.intrinsic_matrix = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]

        # initialize the AprilTag detector
        self.detector = apriltag.Detector()
        # 3D coordinates of the AprilTag corners (in the tag's local coordinate system)
        tag_size = 0.1651  # tag size in meters
        self.tag_3d_points = np.array([[-tag_size/2, -tag_size, 0],
                                       [ tag_size/2, -tag_size, 0],
                                       [ tag_size/2, 0, 0],
                                       [-tag_size/2, 0, 0]], dtype=np.float32)
        
        # various transformation data plus locks for multithreading
        self.transform_tag_to_robot = None
        self.transform_lock = Lock()
        self.gripper_pose = None
        self.gripper_pose_lock = Lock()
        self.position = None

        # trajectory data
        self.trajectory_index = None
        self.trajectory = None

        # serial port to send gripper position to arm (not ideal since everything else can work with serial or UDP)
        self.serial_port = serial.Serial(serial_port_name, baudrate=115200, parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE, timeout=0)
        
        # start a separate thread for image processing
        self.processing_thread = Thread(target=self.update_transform_loop)
        self.processing_thread.daemon = True
    
    # request a trajectory based on initial position
    def send_initial_position(self):
        # detect AprilTags in RTSP stream
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Error: Couldn't read frame from stream.")
            rclpy.shutdown()
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)

        for tag in tags:
            corners = tag.corners

            # estimate the pose of the tag
            image_points = np.array(corners, dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(self.tag_3d_points, image_points, self.intrinsic_matrix, self.dist_coeffs)

            if not success:
                continue

            # convert to robot's coordinate system and account for mounted camera offset and hook offset
            tvec = np.array([tvec[2] - (0.026 + 0.120), -tvec[0], -tvec[1] + 0.033])

            # get camera position in AprilTag frame
            R, _ = cv2.Rodrigues(rvec)
            camera_position_in_tag = -np.transpose(R) @ tvec

            # convert to Point object and publish
            pos = camera_position_in_tag.flatten()
            initial_position = Point()
            initial_position.x = float(pos[0])
            initial_position.y = float(pos[1])
            initial_position.z = float(pos[2])
            self.request_publisher.publish(initial_position)
            self.request_timer.cancel()
            self.get_logger().info(f"Requested trajectory with initial position {initial_position}.")

            return

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
        
        self.trajectory_index = 0
        self.get_logger().info(f"Trajectory contains {len(self.trajectory)} points.")
        
        # start a timer to publish each trajectory step
        self.processing_thread.start()
        self.step_timer = self.create_timer(self.time_step, self.next_traj_step)

    def next_traj_step(self):
        with self.transform_lock:
            transform_tag_to_robot = self.transform_tag_to_robot

        # only proceed if the transform has been initialized
        if transform_tag_to_robot is None:
            return

        if self.trajectory_index < len(self.trajectory):
            # transform the target hook position to the robot's coordinate frame
            gripper_target_position_tag = np.array([self.trajectory[self.trajectory_index].x,
                                                    self.trajectory[self.trajectory_index].y,
                                                    self.trajectory[self.trajectory_index].z,
                                                    1])
            gripper_target_position_robot = np.dot(transform_tag_to_robot, gripper_target_position_tag)
            target_position_gripper_in_robot = gripper_target_position_robot[:3]

            self.publish_pose(target_position_gripper_in_robot[0], target_position_gripper_in_robot[1], target_position_gripper_in_robot[2], gripper_open=False)
        elif self.trajectory_index < len(self.trajectory) + 5:
            # transform the target gripper position to the robot's coordinate frame
            gripper_target_position_tag = np.array([self.trajectory[-1].x,
                                                    self.trajectory[-1].y,
                                                    self.trajectory[-1].z,
                                                    1])
            gripper_target_position_robot = np.dot(transform_tag_to_robot, gripper_target_position_tag)
            target_position_gripper_in_robot = gripper_target_position_robot[:3]

            self.publish_pose(target_position_gripper_in_robot[0], target_position_gripper_in_robot[1], target_position_gripper_in_robot[2], gripper_open=True)
        elif self.trajectory_index < len(self.trajectory) + 10:
            # return arm to ready pose
            self.publish_pose(0.000, 0.024 + 0.120, 0.179, gripper_open=False, set_position=True)
        else:
            self.step_timer.cancel()
        
        # increment before next call
        self.trajectory_index += 1
    
    def publish_pose(self, x, y, z, gripper_open, set_position=True):
        # yaw rotation matrix (point away from origin)
        yaw = np.array([x, y])
        yaw /= np.linalg.norm(yaw)
        R_yaw = np.array([
            [yaw[0], -yaw[1], 0],
            [yaw[1],  yaw[0], 0],
            [0, 0, 1]
        ])
        
        # get most recent gripper orientation
        with self.gripper_pose_lock:
            q = self.gripper_pose.orientation
        q = [q.x, q.y, q.z, q.w]

        # pitch rotation matrix (use previous position)
        _, pitch, _ = R.from_quat(q).as_euler('xyz')
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # combined the rotation matrix
        rotation_matrix = R_yaw @ R_pitch

        # translate hook position backward to end effector position
        offset_vector = np.array([-0.120, 0, 0]) # hook offset from end effector in meters
        rotated_offset = rotation_matrix @ offset_vector
        x += rotated_offset[0]
        y += rotated_offset[1]
        z += rotated_offset[2]

        # update position
        max_movement_speed = 0.001
        movement_vector = np.array([x - self.position.x, y - self.position.y, z - self.position.z])
        magnitude = np.linalg.norm(movement_vector)
        if not set_position and magnitude > max_movement_speed:
            movement_vector *= max_movement_speed / magnitude
        # if magnitude < 0.01:
        #     self.trajectory_index += 1
        self.position.x += movement_vector[0]
        self.position.y += movement_vector[1]
        self.position.z += movement_vector[2]

        # timestamp and publish pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "alpha_base_link"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = self.position.x
        pose_stamped.pose.position.y = self.position.y
        pose_stamped.pose.position.z = self.position.z
        self.pose_publisher.publish(pose_stamped)
        self.get_logger().info(f"Published trajectory step {self.trajectory_index + 1} at xyz {pose_stamped.pose.position.x} {pose_stamped.pose.position.y} {pose_stamped.pose.position.z}.")

        # set gripper joint positions
        if gripper_open:
            gripper_position = [3.7, 2.10] # gripper open
        else:
            gripper_position = [3.35, 2.10] # gripper closed
        packets = b''
        for index, position in enumerate(gripper_position):
            device_id = index + 1
            packets += BPLProtocol.encode_packet(device_id, PacketID.POSITION, BPLProtocol.encode_floats([position]))
        self.serial_port.write(packets)

    def pose_callback(self, msg):
        # avoid logging when gripper position is at origin (bug check)
        if msg.pose.position.x == 0.0 and msg.pose.position.y == 0.0 and msg.pose.position.z == 0.0:
            return
        
        with self.gripper_pose_lock:
            self.gripper_pose = msg.pose
        
    # get end effector pose to update AprilTag-to-robot transform
    def update_transform_loop(self):
        while rclpy.ok():
            # detect AprilTags in RTSP stream
            ret, frame = self.cap.read()
            if not ret:
                rclpy.shutdown()
                return
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = self.detector.detect(gray)

            # get most recent gripper pose in robot frame
            with self.gripper_pose_lock:
                gripper_pose = self.gripper_pose
            if gripper_pose is None:
                continue
            
            # set initial pose
            if self.position is None:
                self.position = gripper_pose.position

            # compute the transformation from the robot's base frame to the AprilTag's frame
            for tag in tags:
                self.update_transform(tag, gripper_pose)

    def update_transform(self, tag, gripper_pose):
        corners = tag.corners

        # estimate the pose of the tag
        image_points = np.array(corners, dtype=np.float32)
        success, tag_rotation_vector, tag_translation_vector = cv2.solvePnP(
            self.tag_3d_points, image_points, self.intrinsic_matrix, self.dist_coeffs
        )

        if not success:
            return

        # convert to robot's coordinate system and account for mounted camera offset
        tag_translation_vector = np.array([tag_translation_vector[2] - 0.026, -tag_translation_vector[0], -tag_translation_vector[1] + 0.033])

        # transform AprilTag pose into the robot's base frame
        gripper_quat = [gripper_pose.orientation.x, gripper_pose.orientation.y, gripper_pose.orientation.z, gripper_pose.orientation.w]
        R_cam_to_robot = R.from_quat(gripper_quat).as_matrix()
        T_cam_to_robot = np.array([gripper_pose.position.x, gripper_pose.position.y, gripper_pose.position.z])
        transform_cam_to_robot = np.hstack([R_cam_to_robot, T_cam_to_robot.reshape(3, 1)])
        transform_cam_to_robot = np.vstack([transform_cam_to_robot, np.array([0, 0, 0, 1])])

        R_tag_to_cam, _ = cv2.Rodrigues(tag_rotation_vector)
        T_tag_to_cam = tag_translation_vector
        transform_tag_to_cam = np.hstack([R_tag_to_cam, T_tag_to_cam.reshape(3, 1)])
        transform_tag_to_cam = np.vstack([transform_tag_to_cam, np.array([0, 0, 0, 1])])

        transform_tag_to_robot = np.dot(transform_cam_to_robot, transform_tag_to_cam)

        with self.transform_lock:
            self.transform_tag_to_robot = transform_tag_to_robot

    def __del__(self):
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = SkillPerformer()
    rclpy.spin(node)

    # Clean shutdown
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

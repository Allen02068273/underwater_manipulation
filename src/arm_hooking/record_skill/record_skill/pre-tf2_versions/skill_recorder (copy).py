import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import csv

import cv2
import apriltag
import numpy as np
from scipy.spatial.transform import Rotation as R

class SkillRecorder(Node):

    def __init__(self):
        super().__init__('skill_recorder')

        # subscriber to the end_effector_pose topic
        self.subscription = self.create_subscription(
            PoseStamped,
            'end_effector_pose',
            self.pose_callback,
            20  # Queue size
        )

        # define the GStreamer pipeline for low-latency RTSP stream
        rtsp_url = "rtsp://admin:@192.168.2.10:554/stream=1"
        self.gst_pipeline = f"rtspsrc location={rtsp_url} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"

        # initialize OpenCV with the GStreamer pipeline
        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)

        # check if the video capture was successfully opened
        if not self.cap.isOpened():
            self.get_logger().error("Error: Couldn't open the RTSP stream.")
            rclpy.shutdown()
            return None

        # initialize the AprilTag detector
        self.detector = apriltag.Detector()

        # Load camera calibration data
        with np.load("data/camera_calibration.npz") as data:
            self.intrinsic_matrix = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]

        # 3D coordinates of the AprilTag corners (in the tag's local coordinate system)
        tag_size = 0.1651  # tag size in meters
        self.tag_3d_points = np.array([[-tag_size/2, -tag_size, 0],
                                       [ tag_size/2, -tag_size, 0],
                                       [ tag_size/2, 0, 0],
                                       [-tag_size/2, 0, 0]], dtype=np.float32)
        
        # store last known transformation from from robot frame to AprilTag frame
        self.transform_robot_to_tag = None

        # set up CSV file
        csv_filename = "data/skill_data.csv"
        self.file = open(csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.csv_writer.writerow(['timestamp', 'gripper_x', 'gripper_y', 'gripper_z'])

        # debugging CSV file (gripper and AprilTag in robot frame)
        csv_filename_debug = "data/skill_data_debug.csv"
        self.file_debug = open(csv_filename_debug, mode='w', newline='')
        self.csv_writer_debug = csv.writer(self.file_debug)
        self.csv_writer_debug.writerow(['timestamp', 'gripper_x', 'gripper_y', 'gripper_z', 'target_x', 'target_y', 'target_z'])

    def pose_callback(self, msg):
        # get end effector pose and current timestamp
        timestamp = self.get_clock().now().to_msg().sec
        gripper_pose = msg.pose

        # avoid logging when gripper position is at origin (bug check)
        if gripper_pose.position.x == 0.0 and gripper_pose.position.y == 0.0 and gripper_pose.position.z == 0.0:
            return

        # detect AprilTags in RTSP stream
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Error: Couldn't read frame from stream.")
            rclpy.shutdown()
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # AprilTag detection requires grayscale
        tags = self.detector.detect(gray)

        transform_tag_to_robot = None  # used for debugging
        for tag in tags:
            if self.transform_robot_to_tag is None: # remove this line to continuously localize
                transform_tag_to_robot = self.update_transform(tag, gripper_pose)
        
        # only proceed if the transform has been initialized
        if self.transform_robot_to_tag is None:
            return
        
        # translate end effector position forward to hook position
        q = [gripper_pose.orientation.x, gripper_pose.orientation.y, gripper_pose.orientation.z, gripper_pose.orientation.w]
        offset_vector = np.array([0.120, 0, 0]) # hook offset from end effector in meters
        rotated_offset = R.from_quat(q).as_matrix() @ offset_vector
        gripper_pose.position.x += rotated_offset[0]
        gripper_pose.position.y += rotated_offset[1]
        gripper_pose.position.z += rotated_offset[2]

        # transform the gripper position to the tag's coordinate frame
        gripper_position_robot = np.array([gripper_pose.position.x, gripper_pose.position.y, gripper_pose.position.z, 1])
        gripper_position_tag = np.dot(self.transform_robot_to_tag, gripper_position_robot)
        position_gripper_in_tag = gripper_position_tag[:3]

        # log pose to CSV
        self.csv_writer.writerow([
            timestamp,
            position_gripper_in_tag[0],
            position_gripper_in_tag[1],
            position_gripper_in_tag[2]
        ])

        # debugging
        position_tag_in_robot = [None, None, None]
        if transform_tag_to_robot is not None:
            position_tag_in_robot = transform_tag_to_robot[:3, 3]

        # log gripper and tag positions in robot frame to debugging CSV
        self.csv_writer_debug.writerow([
            timestamp,
            gripper_pose.position.x,
            gripper_pose.position.y,
            gripper_pose.position.z,
            # gripper_pose.position.x - rotated_offset[0], # actual gripper position (i.e. not including hook offset)
            # gripper_pose.position.y - rotated_offset[1],
            # gripper_pose.position.z - rotated_offset[2]
            position_tag_in_robot[0],
            position_tag_in_robot[1],
            position_tag_in_robot[2]
        ])
    
    # compute the transformation from the robot's base frame to the AprilTag's frame
    def update_transform(self, tag, gripper_pose):
        corners = tag.corners

        # estimate the pose of the tag
        image_points = np.array(corners, dtype=np.float32)
        success, tag_rotation_vector, tag_translation_vector = cv2.solvePnP(
            self.tag_3d_points, image_points, self.intrinsic_matrix, self.dist_coeffs
        )

        if not success:
            return None

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

        # compute the inverse transformation (robot to tag) and update the stored value
        self.transform_robot_to_tag = np.linalg.inv(transform_tag_to_robot)

        return transform_tag_to_robot  # return for debugging

    def __del__(self):
        self.file.close()
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = SkillRecorder()
    rclpy.spin(node)

    # Clean shutdown
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


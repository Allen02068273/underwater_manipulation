import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import csv

import cv2
import apriltag
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class SkillRecorder(Node):

    def __init__(self):
        super().__init__('skill_recorder')

        # Subscriber to the end_effector_pose topic
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
            exit()

        # initialize the AprilTag detector
        self.detector = apriltag.Detector()

        # camera intrinsic parameters (estimated, contact Reach for clarification if necessary)
        focal_length = (math.sqrt(640**2 + 360**2)/2) / math.tan(math.radians(130/2))  # focal length in pixels for 640x360 image, 130 fov 
        center = (640/2, 360/2)  # principal point (cx, cy) (center of a 640x360 image)
        self.intrinsic_matrix = np.array([[focal_length, 0, center[0]],
                                          [0, focal_length, center[1]],
                                          [0, 0, 1]], dtype=np.float32)

        # 3D coordinates of the AprilTag corners (in the tag's local coordinate system)
        tag_size = 0.1651  # tag size in meters
        self.tag_3d_points = np.array([[-tag_size/2, -tag_size/2, 0], 
                                       [tag_size/2, -tag_size/2, 0], 
                                       [tag_size/2, tag_size/2, 0], 
                                       [-tag_size/2, tag_size/2, 0]], dtype=np.float32)

        # Set up CSV file
        self.csv_filename = "skill_data.csv"
        self.file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.csv_writer.writerow(['timestamp', 'gripper_x', 'gripper_y', 'gripper_z', 'gripper_open', 'target_x', 'target_y', 'target_z'])

    def pose_callback(self, msg):
        # get end effector pose and current timestamp
        timestamp = self.get_clock().now().to_msg().sec
        gripper_pose = msg.pose

        # detect AprilTags in RTSP stream
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Error: Couldn't read frame from stream.")
            exit()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # AprilTag detection requires grayscale
        tags = self.detector.detect(gray)

        position_tag_in_robot = [None, None, None]
        
        # Draw bounding boxes around detected tags
        for tag in tags:
            corners = tag.corners
            for i in range(4):
                # Convert corner points to integers
                pt1 = tuple(map(int, corners[i]))
                pt2 = tuple(map(int, corners[(i+1) % 4]))
                
                # Draw the lines around the tag
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # display the tag's ID and center
            cv2.putText(frame, f"ID: {tag.tag_id}", (int(tag.center[0]), int(tag.center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 2D points (image coordinates of the tag's corners)
            image_points = np.array(corners, dtype=np.float32)

            # estimate the pose of the tag
            success, tag_rotation_vector, tag_translation_vector = cv2.solvePnP(self.tag_3d_points, image_points, self.intrinsic_matrix, None)

            if not success:
                continue

            # put coordinates in robots's coordinate system
            # Reach Alpha coordinate system: +x forward, +y left, +z up
            # Camera coordinate system: +x right, +y down, +z forward
            tag_translation_vector = np.array([tag_translation_vector[2], -tag_translation_vector[0], -tag_translation_vector[1]])

            # transform AprilTag pose into the robot's base link frame
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
            position_tag_in_robot = transform_tag_to_robot[:3, 3]
            rotation_tag_in_robot, _ = cv2.Rodrigues(transform_tag_to_robot[:3, :3])
        
        cv2.imshow('AprilTag Detection', frame)

        # bug occurs sometimes logging gripper position at origin
        if not (gripper_pose.position.x == 0.0 and gripper_pose.position.y == 0.0 and gripper_pose.position.z == 0.0):
            # log pose to CSV
            self.csv_writer.writerow([
                timestamp,
                gripper_pose.position.x,
                gripper_pose.position.y,
                gripper_pose.position.z,
                None,
                position_tag_in_robot[0],
                position_tag_in_robot[1],
                position_tag_in_robot[2]
            ])

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


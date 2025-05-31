import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

import cv2
import apriltag
import numpy as np
from scipy.spatial.transform import Rotation as R

from threading import Thread, Lock

class AprilTagPose(Node):

    def __init__(self):
        super().__init__('apriltag_pose')

        # gripper pose and lock for multi-threaded access
        self.gripper_pose = None
        self.gripper_pose_lock = Lock()

        # subscriber to the end_effector_pose topic
        self.subscription = self.create_subscription(
            PoseStamped,
            'end_effector_pose',
            self.pose_callback,
            10  # Queue size
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

        # load camera calibration data
        with np.load("data/camera_calibration.npz") as data:
            self.intrinsic_matrix = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]

        # 3D coordinates of the AprilTag corners (in the tag's local coordinate system)
        tag_size = 0.1651  # tag size in meters
        self.tag_3d_points = np.array([[-tag_size/2, -tag_size, 0],
                                       [ tag_size/2, -tag_size, 0],
                                       [ tag_size/2, 0, 0],
                                       [-tag_size/2, 0, 0]], dtype=np.float32)
        
        # start a separate thread for image processing
        self.processing_thread = Thread(target=self.image_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def pose_callback(self, msg):
        # avoid logging when gripper position is at origin (bug check)
        if msg.pose.position.x == 0.0 and msg.pose.position.y == 0.0 and msg.pose.position.z == 0.0:
            return
        
        with self.gripper_pose_lock:
            self.gripper_pose = msg.pose
            
    def image_processing_loop(self):
        while rclpy.ok():
            # detect AprilTags in RTSP stream
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("Error: Couldn't read frame from stream.")
                rclpy.shutdown()
                return None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # AprilTag detection requires grayscale
            tags = self.detector.detect(gray)

            # get the most recent end effector pose
            with self.gripper_pose_lock:
                gripper_pose = self.gripper_pose
            if gripper_pose is None:
                continue

            transform_tag_to_robot = None
            for tag in tags:
                corners = tag.corners

                # estimate the pose of the tag
                image_points = np.array(corners, dtype=np.float32)
                success, tag_rotation_vector, tag_translation_vector = cv2.solvePnP(
                    self.tag_3d_points, image_points, self.intrinsic_matrix, self.dist_coeffs
                )

                if not success:
                    continue

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

                break

            # Overlay text on the camera feed
            if transform_tag_to_robot is not None:
                position_tag_in_robot = transform_tag_to_robot[:3, 3]
                font = cv2.FONT_HERSHEY_SIMPLEX
                position_text = f"Position: x={position_tag_in_robot[0]:.3f}, y={position_tag_in_robot[1]:.3f}, z={position_tag_in_robot[2]:.3f}"
                cv2.putText(frame, position_text, (10, 30), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            # Display the image with the overlayed text
            cv2.imshow("Camera Feed with Position", frame)
            cv2.waitKey(1)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


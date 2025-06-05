import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped

import cv2
import apriltag
import numpy as np
from scipy.spatial.transform import Rotation

from ra_core.utils.video import Video

class AprilTagPosePublisher(Node):

    def __init__(self):
        super().__init__('apriltag_pose_publisher')

        self.declare_parameter('apriltag_size', 0.0745)  # meters
        self.declare_parameter('camera_tf_frame', 'reach_alpha_camera')
        self.declare_parameter('frequency', 30)  # Hz
        self.declare_parameter('display_feed', False)

        self.tag_size = float(self.get_parameter('apriltag_size').value)
        self.camera_tf_frame = self.get_parameter('camera_tf_frame').value
        frequency = float(self.get_parameter('frequency').value)
        self.display_feed = self.get_parameter('display_feed').value

        # publisher and timer for publishing AprilTag pose
        self.pose_publisher = self.create_publisher(TransformStamped, 'apriltag_pose', 10)
        self.static_tf_broadcast_timer = self.create_timer(1.0/frequency, self.process_apriltags)

        camera_calibration_path = None
        
        # GStreamer pipeline for low-latency stream
        video_source = None
        if self.camera_tf_frame == 'reach_alpha_camera':
            rtsp_url = "rtsp://admin:@192.168.2.10:554/stream=1"
            video_source = f'rtspsrc location={rtsp_url}'
            camera_calibration_path = "data/camera_calibrations/camera_calibration_nerve_ra.npz"
        else:
            port = 5600
            video_source = f'udpsrc port={port}'
            camera_calibration_path = "data/camera_calibrations/camera_calibration_brov.npz"
        self.video = Video(video_source)

        # load camera calibration data
        with np.load(camera_calibration_path) as data:
            self.intrinsic_matrix = data["camera_matrix"]
            self.dist_coeffs = data["dist_coeffs"]

        # initialize the AprilTag detector
        self.detector = apriltag.Detector()
        # 3D coordinates of the AprilTag corners (in the tag's local coordinate system)
        self.tag_3d_points = np.array([[-self.tag_size/2, -self.tag_size/2, 0],
                                       [ self.tag_size/2, -self.tag_size/2, 0],
                                       [ self.tag_size/2,  self.tag_size/2, 0],
                                       [-self.tag_size/2,  self.tag_size/2, 0]], dtype=np.float32)
        
        self.get_logger().info(f"Starting AprilTag pose publisher for {self.camera_tf_frame}")
        
    def process_apriltags(self):
        # detect AprilTags in the camera stream
        if not self.video.frame_available():
            return
        frame = self.video.frame()
        
        # estimate the image's timestamp
        stamp = self.get_clock().now() - rclpy.time.Duration(seconds=0.04)

        # detect AprilTags
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)

        # compute the transformation from the robot's base frame to the AprilTag's frame
        for tag in tags:
            self.process_tag(tag, stamp, self.camera_tf_frame, (self.intrinsic_matrix, self.dist_coeffs))

        if self.display_feed:
            cv2.imshow(self.camera_tf_frame, frame)
            cv2.waitKey(1)

    def process_tag(self, tag, stamp, camera_frame, camera_calibration):
        corners = tag.corners

        # estimate the pose of the tag
        image_points = np.array(corners, dtype=np.float32)
        success, tag_rotation_vector, tag_translation_vector = cv2.solvePnP(
            self.tag_3d_points, image_points, *camera_calibration
        )
        if not success:
            return
        
        # convert to robot's coordinate system
        tag_translation_vector = np.array([tag_translation_vector[2], -tag_translation_vector[0], -tag_translation_vector[1]])
        rotation_matrix, _ = cv2.Rodrigues(tag_rotation_vector)
        tag_quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
        tag_quaternion = np.array([tag_quaternion[2], -tag_quaternion[0], -tag_quaternion[1], tag_quaternion[3]])
        
        # convert to a TransformStamped and publish
        tfs = TransformStamped()
        tfs.header.stamp = stamp.to_msg()
        tfs.header.frame_id = camera_frame
        tfs._child_frame_id = f'apriltag_{tag.tag_id}'
        tfs.transform.translation.x = float(tag_translation_vector[0])
        tfs.transform.translation.y = float(tag_translation_vector[1])
        tfs.transform.translation.z = float(tag_translation_vector[2])
        tfs.transform.rotation.x = tag_quaternion[0]
        tfs.transform.rotation.y = tag_quaternion[1]
        tfs.transform.rotation.z = tag_quaternion[2]
        tfs.transform.rotation.w = tag_quaternion[3]
        self.pose_publisher.publish(tfs)

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

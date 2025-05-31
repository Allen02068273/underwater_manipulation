import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped

import cv2
import apriltag
import numpy as np
from scipy.spatial.transform import Rotation

import numpy as np

class AprilTagPosePublisher(Node):

    def __init__(self):
        super().__init__('apriltag_pose_publisher')

        self.declare_parameter('apriltag_size', 0.0745)  # meters
        self.declare_parameter('camera_tf_frame', 'reach_alpha_camera')
        self.declare_parameter('frequency', 30)  # Hz

        self.tag_size = float(self.get_parameter('apriltag_size').value)
        self.camera_tf_frame = self.get_parameter('camera_tf_frame').value
        frequency = float(self.get_parameter('frequency').value)

        # publisher and timer for publishing AprilTag pose
        self.pose_publisher = self.create_publisher(TransformStamped, 'apriltag_pose', 10)
        self.static_tf_broadcast_timer = self.create_timer(1.0/frequency, self.process_apriltags)

        camera_calibration_path = None
        
        # GStreamer pipeline for low-latency stream
        self.gst_pipeline = None
        if self.camera_tf_frame == 'reach_alpha_camera':
            rtsp_url = "rtsp://admin:@192.168.2.10:554/stream=1"
            self.gst_pipeline = f"rtspsrc location={rtsp_url} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
            camera_calibration_path = "data/camera_calibrations/camera_calibration_nerve_ra.npz"
        else:
            udp_port = 5600  # port used by the UDP stream
            self.gst_pipeline = (
                f"udpsrc port={udp_port} "
                "! application/x-rtp, payload=96 "
                "! rtph264depay ! h264parse ! avdec_h264 "
                "! decodebin "
                "! videoconvert ! video/x-raw,format=BGR "
                "! videoconvert ! appsink emit-signals=true sync=false max-buffers=2 drop=true"
            )
            camera_calibration_path = "data/camera_calibrations/camera_calibration_brov.npz"
        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error("Error: Couldn't open the RTSP stream.")
            rclpy.shutdown()
            return None

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
        
    # get end effector pose to update AprilTag-to-robot transform
    def process_apriltags(self):
        # detect AprilTags in the camera stream
        ret, frame = self.cap.read()
        if not ret:
            # rclpy.shutdown()
            return
        
        # estimate the image's timestamp
        stamp = self.get_clock().now() - rclpy.time.Duration(seconds=0.04)

        # detect AprilTags
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)

        # compute the transformation from the robot's base frame to the AprilTag's frame
        for tag in tags:
            self.process_tag(tag, stamp, self.camera_tf_frame, (self.intrinsic_matrix, self.dist_coeffs))

    def process_tag(self, tag, stamp, camera_frame, camera_calibration):
        corners = tag.corners

        # estimate the pose of the tag
        image_points = np.array(corners, dtype=np.float32)
        success, tag_rotation_vector, tag_translation_vector = cv2.solvePnP(
            self.tag_3d_points, image_points, *camera_calibration
        )
        if not success:
            return False
        
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

        return True

    def __del__(self):
        self.cap.release()

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

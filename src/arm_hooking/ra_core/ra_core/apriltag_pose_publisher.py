import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError
import cv2
import apriltag
import numpy as np
from scipy.spatial.transform import Rotation

# from ra_core.utils.video import Video

class AprilTagPosePublisher(Node):

    def __init__(self):
        super().__init__('apriltag_pose_publisher')

        self.declare_parameter('apriltag_size', 0.0745)  # meters
        self.declare_parameter('frequency', 30)  # Hz
        self.declare_parameter('display_feed', False)
        self.declare_parameter('use_gstreamer', False)  # use GStreamer for low-latency RA camera feed
        # note: enabling GStreamer will likely require building OpenCV from source with GStreamer enabled

        self.tag_size = float(self.get_parameter('apriltag_size').value)
        frequency = float(self.get_parameter('frequency').value)
        self.display_feed = self.get_parameter('display_feed').value
        use_gstreamer = self.get_parameter('use_gstreamer').value

        # timer, subscriber, and publisher for processing and publishing AprilTag pose
        self.ra_camera_timer = self.create_timer(1.0/frequency, self.ra_img_processing)
        self.subscription = self.create_subscription(Image, '/my_camera/image_raw', self.brov_img_callback, 10)
        self.pose_publisher = self.create_publisher(TransformStamped, 'apriltag_pose', 10)

        self.bridge = CvBridge()
        
        video_source = "rtsp://admin:@192.168.2.10:554/stream=1"
        if use_gstreamer:  # GStreamer pipeline for low-latency stream
            video_source = f'rtspsrc location={video_source} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink'
            self.cap = cv2.VideoCapture(video_source, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(video_source)

        # cancel the Reach Alpha image processing timer if the camera stream is not open
        if not self.cap.isOpened():
            self.ra_camera_timer.cancel()
            self.get_logger().error(f"Can't open RTSP stream from Reach Alpha camera")

        # load camera calibration data
        with np.load("data/camera_calibrations/camera_calibration_nerve_ra.npz") as data:
            self.ra_camera_calibration = (data["camera_matrix"], data["dist_coeffs"])
        with np.load("data/camera_calibrations/camera_calibration_brov.npz") as data:
            self.brov_camera_calibration = (data["camera_matrix"], data["dist_coeffs"])

        # initialize the AprilTag detector
        self.detector = apriltag.Detector()
        # 3D coordinates of the AprilTag corners (in the tag's local coordinate system)
        self.tag_3d_points = np.array([[-self.tag_size/2, -self.tag_size/2, 0],
                                       [ self.tag_size/2, -self.tag_size/2, 0],
                                       [ self.tag_size/2,  self.tag_size/2, 0],
                                       [-self.tag_size/2,  self.tag_size/2, 0]], dtype=np.float32)
        
    def ra_img_processing(self):
        # read in the next frame from the camera stream
        ret, frame = self.cap.read()
        if not ret:
            self.ra_camera_timer.cancel()
            self.get_logger().error(f"Unable to continue reading RTSP stream from Reach Alpha camera")
            return

        # detect AprilTags
        self.process_frame(frame, 'reach_alpha_camera', self.ra_camera_calibration)

    def brov_img_callback(self, msg):
        # convert ROS Image message to OpenCV image
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

        # detect AprilTags
        self.process_frame(frame, 'reach_alpha_camera', self.brov_camera_calibration)

    def process_frame(self, frame, tf_frame, camera_calibration):
        # estimate the image's timestamp
        stamp = self.get_clock().now() - rclpy.time.Duration(seconds=0.04)
        
        # detect AprilTags
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray)

        # compute the transformation from the camera's frame to the AprilTag's frame
        for tag in tags:
            self.process_tag(tag, stamp, tf_frame, camera_calibration)

        # display the camera frame if requested by the user
        if self.display_feed:
            cv2.imshow(tf_frame, frame)
            cv2.waitKey(1)

    def process_tag(self, tag, stamp, tf_frame, camera_calibration):
        corners = tag.corners

        # estimate the pose of the tag
        image_points = np.array(corners, dtype=np.float32)
        success, tag_rotation_vector, tag_translation_vector = cv2.solvePnP(
            self.tag_3d_points, image_points, *camera_calibration
        )
        if not success:
            return
        
        # convert to ROS2's coordinate system
        tag_translation_vector = np.array([tag_translation_vector[2], -tag_translation_vector[0], -tag_translation_vector[1]])
        rotation_matrix, _ = cv2.Rodrigues(tag_rotation_vector)
        tag_quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
        tag_quaternion = np.array([tag_quaternion[2], -tag_quaternion[0], -tag_quaternion[1], tag_quaternion[3]])
        
        # convert to a TransformStamped and publish
        tfs = TransformStamped()
        tfs.header.stamp = stamp.to_msg()
        tfs.header.frame_id = tf_frame
        tfs._child_frame_id = f'apriltag_{tag.tag_id}'
        tfs.transform.translation.x = float(tag_translation_vector[0])
        tfs.transform.translation.y = float(tag_translation_vector[1])
        tfs.transform.translation.z = float(tag_translation_vector[2])
        tfs.transform.rotation.x = tag_quaternion[0]
        tfs.transform.rotation.y = tag_quaternion[1]
        tfs.transform.rotation.z = tag_quaternion[2]
        tfs.transform.rotation.w = tag_quaternion[3]
        self.pose_publisher.publish(tfs)

def __del__(self):
    cap.release()
    cv2.destroyAllWindows()

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

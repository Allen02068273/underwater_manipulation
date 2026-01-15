import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError
import cv2
from datetime import datetime

class BROVCamRecorder(Node):

    def __init__(self):
        super().__init__('brov_cam_recorder')

        # get the output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.declare_parameter('output_path', f'data/camera_recordings/brov_camera_{timestamp}.avi')
        self.output_path = self.get_parameter('output_path').value

        # subscribe to the camera topic
        self.subscription = self.create_subscription(Image, '/my_camera/image_raw', self.brov_img_callback, 10)

        self.bridge = CvBridge()

        # initialize video writer after first frame (when we know width/height)
        self.video_writer = None
        self.fps = 30.0  # may not be BROV's fps, check later

        self.get_logger().info(f"Recording BROV camera to {self.output_path}")

    def brov_img_callback(self, msg):
        # convert ROS Image message to OpenCV image
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # initialize writer if needed
        if self.video_writer is None:
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MJPG'/'MP4V'
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            if not self.video_writer.isOpened():
                self.get_logger().error("Failed to open VideoWriter")
                return

        # write frame to video file
        self.video_writer.write(frame)

    def __del__(self):
        if self.video_writer is not None:
            self.video_writer.release()


def main(args=None):
    rclpy.init(args=args)
    node = BROVCamRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

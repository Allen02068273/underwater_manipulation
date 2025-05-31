import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from datetime import datetime

class FocusDetection(Node):

    def __init__(self):
        super().__init__('focus_detection')

        # parameters
        self.declare_parameter('threshold', 2000.0)
        self.threshold = float(self.get_parameter('threshold').value)
        self.declare_parameter('record_output', False)
        record_output_param = str(self.get_parameter('record_output').value)
        if isinstance(record_output_param, bool):
            self.record_output = record_output_param
        else:
            self.record_output = str(record_output_param).lower() in ['y', 'yes', 'true', '1']

        # Logging values for debugging
        self.get_logger().info(f"Focus threshold set to: {self.threshold}")
        self.get_logger().info(f"Recording output: {self.record_output}")

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

        # Get video properties
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Initialize video writer if recording
        self.out = None
        if self.record_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f'focus_analysis_{timestamp}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width * 2, frame_height))
        
        # timer for processing and visualizing the camera feed
        self.timer = self.create_timer(1.0/fps, self.image_processing_loop)

        self.get_logger().info("Starting video analysis")
        self.get_logger().info(f"Focus threshold: {self.threshold}")
        self.get_logger().info(f"Recording to {output_filename}" if self.record_output else "Not recording")
        self.get_logger().info("Press 'q' to quit")
            
    def image_processing_loop(self):
        kernel_size=3

        # detect AprilTags in RTSP stream
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Error: Couldn't read frame from stream.")
            rclpy.shutdown()
            return None
        
        # Analyze focus
        is_focused, focus_measure, laplacian = self.check_frame_focus(
            frame, self.threshold, kernel_size
        )

        # Create visualization
        # Original frame
        status_color = (0, 255, 0) if is_focused else (0, 0, 255)  # Green or Red
        status_text = "FOCUSED" if is_focused else "UNFOCUSED"
        
        # Add focus information to frame
        info_frame = frame.copy()
        cv2.putText(info_frame, f"Focus: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(info_frame, f"Measure: {focus_measure:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Normalize Laplacian for visualization
        laplacian_vis = cv2.normalize(np.absolute(laplacian), None, 0, 255, 
                                    cv2.NORM_MINMAX).astype(np.uint8)
        laplacian_vis = cv2.applyColorMap(laplacian_vis, cv2.COLORMAP_HOT)
        
        # Combine original and Laplacian views
        combined_frame = np.hstack((info_frame, laplacian_vis))
        
        # Record if enabled
        if self.record_output:
            self.out.write(combined_frame)
        
        # Display result
        cv2.imshow('Focus Analysis', combined_frame)
        cv2.waitKey(1)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.timer.cancel()
            self.get_logger().info("User quit")
    
    def check_frame_focus(self, frame, threshold=100.0, kernel_size=3):
        # Convert to grayscale if frame is in color
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Compute Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        
        # Calculate variance of Laplacian
        focus_measure = np.var(laplacian)
        
        # Determine if frame is focused based on threshold
        is_focused = focus_measure > threshold
        
        return is_focused, focus_measure, laplacian

    def __del__(self):
        self.cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = FocusDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

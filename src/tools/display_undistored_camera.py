import cv2
import numpy as np

# Load camera calibration data
with np.load("data/camera_calibration.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

# GStreamer pipeline for low-latency RTSP stream
rtsp_url = "rtsp://admin:@192.168.2.10:554/stream=1"
gst_pipeline = f"rtspsrc location={rtsp_url} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# Check if the video capture was successfully opened
if not cap.isOpened():
    print("Error: Couldn't open the RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame from stream.")
        break
    
    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    
    # Concatenate the original and undistorted frames side by side
    side_by_side = np.hstack((frame, undistorted_frame))
    
    # Display the frames
    cv2.imshow("Original vs Undistorted", side_by_side)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

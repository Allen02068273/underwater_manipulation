import cv2
import numpy as np

# GStreamer pipeline for low-latency RTSP stream
rtsp_url = "rtsp://admin:@192.168.2.10:554/stream=1"
gst_pipeline = f"rtspsrc location={rtsp_url} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Couldn't open the RTSP stream.")
    exit()

square_size = 0.022  # 22mm in meters
chessboard_size = (9, 6)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # scale object points to meters

objpoints = []
imgpoints = []

frame_count = 0
required_frames = 20

print("Move the chessboard around and press 's' to capture a frame. Press 'q' to quit.")

while frame_count < required_frames:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame from stream.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        cv2.putText(frame, "Press 's' to save this frame", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Camera Calibration', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        frame_count += 1
        print(f"Frame {frame_count}/{required_frames} captured.")

    if key == ord('q'):
        break

print(f"Collected {frame_count} frames. Performing calibration...")

if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)

    np.savez("camera_calibration.npz", camera_matrix=mtx, dist_coeffs=dist)
    print("Calibration data saved.")

cap.release()
cv2.destroyAllWindows()

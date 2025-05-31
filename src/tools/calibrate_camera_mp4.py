import cv2
import numpy as np

# open the video file
video_file = "data/camera_recordings/pool_chessboard_2.mp4"
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
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

print("Frames captured automatically. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file reached.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        objpoints.append(objp)
        imgpoints.append(corners)
        frame_count += 1
        print(f"Frame {frame_count}/{required_frames} captured.")

    cv2.imshow('Camera Calibration', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
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

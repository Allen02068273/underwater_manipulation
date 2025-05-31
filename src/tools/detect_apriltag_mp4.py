import cv2
import apriltag
import numpy as np

# load camera calibration data
with np.load("data/camera_calibration.npz") as data:
    intrinsic_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

# 3D coordinates of the AprilTag corners (in the tag's local coordinate system)
tag_size = 0.050  # tag size in meters
tag_3d_points = np.array([[-tag_size/2, -tag_size, 0],
                          [ tag_size/2, -tag_size, 0],
                          [ tag_size/2, 0, 0],
                          [-tag_size/2, 0, 0]], dtype=np.float32)

# open the video file
video_file = "data/camera_recordings/pool_apriltag_50mm.mp4"
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

# initialize the AprilTag detector
detector = apriltag.Detector()

# track various AprilTag detection data
max_depth = -np.inf
max_depth_frame = None
min_depth = np.inf
min_depth_frame = None
depth = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file reached.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    for tag in tags:
        corners = tag.corners

        # draw lines around the tag
        for i in range(4):
            pt1 = tuple(map(int, corners[i]))
            pt2 = tuple(map(int, corners[(i+1) % 4]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # estimate the pose of the tag
        image_points = np.array(corners, dtype=np.float32)
        success, tag_rotation_vector, tag_translation_vector = cv2.solvePnP(
            tag_3d_points, image_points, intrinsic_matrix, dist_coeffs
        )

        if not success:
            continue

        # track various AprilTag detection data
        depth = tag_translation_vector[2]
        if depth > max_depth:
            max_depth = depth
            max_depth_frame = frame.copy()
        if depth < min_depth:
            min_depth = depth
            min_depth_frame = frame.copy()

        break
    
    # display the resulting frame with AprilTags annotated
    cv2.imshow('AprilTag Detection', frame)
    cv2.waitKey(1)

    # exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# save and display the max and min recorded depth and associated frames
if max_depth_frame is not None:
    print(f"Maximum depth: {max_depth} meters")
    print(f"Minimum depth: {min_depth} meters")
    cv2.imwrite("data/farthest_50mm_apriltag.png", max_depth_frame)
    cv2.imwrite("data/closest_50mm_apriltag.png", min_depth_frame)
    cv2.imshow('Max Depth Frame', max_depth_frame)
    cv2.imshow('Min Depth Frame', min_depth_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No AprilTags were detected in the video.")

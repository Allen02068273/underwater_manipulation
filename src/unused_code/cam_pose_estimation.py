import cv2
import apriltag
import numpy as np
import math

# Define the GStreamer pipeline for low-latency RTSP stream
rtsp_url = "rtsp://admin:@192.168.2.10:554/stream=1"
gst_pipeline = f"rtspsrc location={rtsp_url} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"

# Initialize OpenCV with the GStreamer pipeline
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# Check if the video capture was successfully opened
if not cap.isOpened():
    print("Error: Couldn't open the RTSP stream.")
    exit()

# Initialize the AprilTag detector
detector = apriltag.Detector()

# Camera intrinsic parameters (estimated, contact Reach for clarification if necessary)
focal_length = (math.sqrt(640**2 + 360**2)/2) / math.tan(math.radians(130/2))  # focal length in pixels for 640x360 image, 130 fov 
center = (640/2, 360/2)  # principal point (cx, cy) (center of a 640x360 image)
intrinsic_matrix = np.array([[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype=np.float32)

# Known 3D coordinates of the AprilTag corners (in the tag's local coordinate system)
tag_size = 0.1651  # tag size in meters

# 3D points in the tag's local coordinate system
tag_3d_points = np.array([[-tag_size/2, -tag_size/2, 0], 
                          [tag_size/2, -tag_size/2, 0], 
                          [tag_size/2, tag_size/2, 0], 
                          [-tag_size/2, tag_size/2, 0]], dtype=np.float32)

while True:
    # Capture frame-by-frame from the RTSP stream
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Couldn't read frame from stream.")
        break
    
    # Convert the frame to grayscale for AprilTag detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect AprilTags in the frame
    tags = detector.detect(gray)
    
    # Draw bounding boxes around detected tags
    for tag in tags:
        corners = tag.corners
        for i in range(4):
            # Convert corner points to integers
            pt1 = tuple(map(int, corners[i]))
            pt2 = tuple(map(int, corners[(i+1) % 4]))
            
            # Draw the lines around the tag
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Optionally, display the tag's ID and center
        cv2.putText(frame, f"ID: {tag.tag_id}", (int(tag.center[0]), int(tag.center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 2D points (image coordinates of the tag's corners)
        image_points = np.array(corners, dtype=np.float32)

        # Use solvePnP to estimate the pose of the tag
        success, rotation_vector, translation_vector = cv2.solvePnP(tag_3d_points, image_points, intrinsic_matrix, None)
        
        if success:
            # Project the 3D corners back to 2D space
            projected_points, _ = cv2.projectPoints(tag_3d_points, rotation_vector, translation_vector, intrinsic_matrix, None)
            projected_points = projected_points.reshape(-1, 2)
            
            # Draw the projected outline of the tag
            for i in range(4):
                pt1 = tuple(map(int, projected_points[i]))
                pt2 = tuple(map(int, projected_points[(i+1) % 4]))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)  # Red outline for projected corners
            
            # Optionally print the 3D translation vector (position of the tag in 3D space)
            print(f"Tag {tag.tag_id} position (in meters): {translation_vector.ravel()}")

    # Display the resulting frame with AprilTags detected
    cv2.imshow('AprilTag Detection', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

import cv2
import apriltag

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
    
    # Display the resulting frame with AprilTags detected
    cv2.imshow('AprilTag Detection', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

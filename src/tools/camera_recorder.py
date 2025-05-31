import cv2
import numpy as np
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Display and record Reach Alpha camera feed.")
parser.add_argument('--output', type=str, default=None, 
                    help='Output video file name (default: data/camera_recordings/reach_alpha_camera_feed_[timestamp].mp4)')
args = parser.parse_args()

if args.output:
    output_file = args.output
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/camera_recordings/reach_alpha_camera_feed_{timestamp}.mp4"  # default file

# GStreamer pipeline for low-latency RTSP stream
rtsp_url = "rtsp://admin:@192.168.2.10:554/stream=1"
gst_pipeline = f"rtspsrc location={rtsp_url} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Couldn't open the RTSP stream.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:  # fallback if FPS is not obtained from stream
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame from stream.")
        break
    
    cv2.imshow("Reach Alpha Camera Feed", frame)
    
    out.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # exit if 'q' is pressed
        break

cap.release()
out.release()
cv2.destroyAllWindows()

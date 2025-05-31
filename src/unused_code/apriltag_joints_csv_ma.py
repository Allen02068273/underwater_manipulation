# apriltag_joints_csv.py
# by Nolan Allen
# Copyright 2025
# adapted from https://github.com/Reach-Robotics/reach_robotics_sdk/blob/master/bplprotocol/examples/joint_information.py
# and https://github.com/Reach-Robotics/reach_robotics_sdk/blob/master/bplprotocol/examples/master_arm_passthrough_serial.py

import threading

# reach alpha imports
from bplprotocol import BPLProtocol, PacketID, PacketReader
import time
import serial # install pyserial with pip install pyserial

# master arm imports
import time
import serial # install pyserial with pip install pyserial

# apriltag imports
import cv2
import apriltag

# csv imports
import csv


# reach alpha setup
packet_reader = PacketReader()
ARM_PORT = "/dev/ttyUSB0"
request_timeout = 0.5  # Seconds
arm_serial = serial.Serial(ARM_PORT, baudrate=115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0)
device_id = 0x01  # Jaws
start_time = 0


# master arm setup and thread
MASTER_ARM_PORT = "/dev/ttyUSB1"
master_arm_serial = serial.Serial(MASTER_ARM_PORT, baudrate=115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0)
def master_arm_passthrough():
    global master_arm_serial, arm_serial
    print("Beginning passthrough from master arm to arm")
    while True:
        time.sleep(0.0001)
        # send master arm commands to reach alpha
        try:
            data = master_arm_serial.read()
        except BaseException as e:
            print(f"Error Reading from master arm serial port: {e}")
            data = b''

        if data != b'':
            try:
                arm_serial.write(data)
            except BaseException as e:
                print(f"Error Writing to an arms serial port: {e}")

master_arm_thread = threading.Thread(target=master_arm_passthrough, daemon=True)
master_arm_thread.start()


# arm camera setup
rtsp_url = "rtsp://admin:@192.168.2.10:554/stream=1"
gst_pipeline = f"rtspsrc location={rtsp_url} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Error: Couldn't open the RTSP stream.")
    exit()

detector = apriltag.Detector()


# csv setup
csv_file_path = 'data.csv'
fieldnames = ['arm_pos', 'tag_pos']
with open(csv_file_path, mode='w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

arm_flag = False
arm_pos = None
tag_flag = False
tag_pos = 0

try:
    # main program loop
    while True:
        # process jaw position
        if not arm_flag and time.time() - start_time > request_timeout:
            # Request POSITION from the jaws after timeout or after position received
            arm_serial.write(BPLProtocol.encode_packet(device_id, PacketID.REQUEST, bytes([PacketID.POSITION])))
            start_time = time.time()
        try:
            read_data = arm_serial.read()
        except BaseException:
            read_data = b''
        if read_data != b'':
            packets = packet_reader.receive_bytes(read_data)
            if packets:
                for packet in packets:
                    read_device_id, read_packet_id, data_bytes = packet
                    if read_device_id == device_id and read_packet_id == PacketID.POSITION:
                        # Decode floats, because position is reported in floats
                        arm_pos = BPLProtocol.decode_floats(data_bytes)[0]
                        print(f"Position from Device {device_id} is {arm_pos}")
                if arm_pos is not None:
                    arm_flag = True
                    start_time = 0
        
        # process a frame from the camnera feed
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame from stream.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # detector requires grayscale
        tags = detector.detect(gray)
        tag_pos = None
        tag_flag = True
        for tag in tags:
            tag_pos = (int(tag.center[0]), int(tag.center[1]))
        
        # display bounding boxes and tag data (can remove later)
        for tag in tags:
            corners = tag.corners
            for i in range(4):
                pt1 = tuple(map(int, corners[i]))
                pt2 = tuple(map(int, corners[(i+1) % 4]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {tag.tag_id}", (int(tag.center[0]), int(tag.center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('AprilTag Detection', frame)
        
        if arm_flag and tag_flag:
            arm_flag = False
            tag_flag = False
            # write the data to the csv file
            with open(csv_file_path, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'arm_pos': arm_pos,
                    'tag_pos': tag_pos
                })
        
        # quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopped by user.")


# clean up
cap.release()
cv2.destroyAllWindows()

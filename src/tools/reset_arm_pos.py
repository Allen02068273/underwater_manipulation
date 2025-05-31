from bplprotocol import BPLProtocol, PacketID
import serial
import argparse

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run the program with a specified serial port to connect to the manipulator.")
    parser.add_argument(
        "--serial_port", 
        type=str, 
        default="/dev/ttyUSB0", 
        help="Serial port to connect to the manipulator (default: /dev/ttyUSB0)"
    )

    # Parse arguments
    args = parser.parse_args()
    serial_port_name = args.serial_port

    serial_port = serial.Serial(serial_port_name, baudrate=115200, parity=serial.PARITY_NONE,
                                stopbits=serial.STOPBITS_ONE, timeout=0)

    # Starting position, end effector to base A -> E

    # # docked pose
    # desired_positions = [3.35,
    #                      2.10,
    #                      0.00,
    #                      1.59,
    #                      1.57]
    
    # ready pose
    desired_positions = [3.35,
                         2.10,
                         0.60,
                         2.23,
                         1.57]


    packets = b''
    for index, position in enumerate(desired_positions):
        device_id = index + 1
        packets += BPLProtocol.encode_packet(device_id, PacketID.POSITION, BPLProtocol.encode_floats([position]))

    # Send joints to desired_positions
    serial_port.write(packets)

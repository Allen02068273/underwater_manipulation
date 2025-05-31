from bplprotocol import BPLProtocol, PacketID, PacketReader

import time

# install pyserial with pip install pyserial
import serial


if __name__ == '__main__':

    packet_reader = PacketReader()

    serial_port_name = "/dev/ttyUSB0"

    request_timeout = 0.5  # Seconds

    serial_port = serial.Serial(serial_port_name, baudrate=115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0)

    for i in range(5):
        # Requesting Information
        device_id = i+1

        print(f"Requesting Position from Device {device_id}")

        # Request POSITION from the jaws
        serial_port.write(BPLProtocol.encode_packet(device_id, PacketID.REQUEST, bytes([PacketID.POSITION])))

        start_time = time.time()

        position = None
        while True:
            time.sleep(0.0001)
            try:
                read_data = serial_port.read()
            except BaseException:
                read_data = b''
            if read_data != b'':
                packets = packet_reader.receive_bytes(read_data)
                if packets:
                    for packet in packets:
                        read_device_id, read_packet_id, data_bytes = packet
                        if read_device_id == device_id and read_packet_id == PacketID.POSITION:

                            # Decode floats, because position is reported in floats
                            position = BPLProtocol.decode_floats(data_bytes)[0]
                            print(f"Position from Device {device_id} is {position}")

                    if position is not None:
                        break

            # Timeout if no response is seen from the device.
            if time.time() - start_time > request_timeout:
                print("Request for Position timed out")
                break

        time.sleep(0.1)
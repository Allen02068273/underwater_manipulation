import rclpy
from rclpy.node import Node

from bpl_msgs.msg import Packet

from bplprotocol import PacketReader

import serial

class MasterArmControl(Node):
    def __init__(self):
        super().__init__('master_arm_control')

        # serial port for the Master Arm
        self.declare_parameter('master_arm_serial_port', '/dev/ttyUSB1')
        master_arm_port = self.get_parameter('master_arm_serial_port').value
        self.master_arm_serial = serial.Serial(master_arm_port, baudrate=115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0)
        self.packet_reader = PacketReader()
        self.get_logger().info(f"Opened serial port {master_arm_port}")

        # publish Master Arm data to BPL Passthrough
        self.tx_publisher = self.create_publisher(Packet, "tx", 10)
        self.frequency = 100  # Hz
        self.create_timer(1.0/self.frequency, self.read_serial)

    def read_serial(self):
        try:
            data = self.master_arm_serial.read(1024)
        except serial.SerialException as e:
            print(f"Error reading from master arm serial port: {e}")
            return
        
        if data != b'':
            packets = self.packet_reader.receive_bytes(data)

            for packet in packets:
                device_id, packet_id, data_bytes = packet
                p = Packet()
                p.packet_id = packet_id
                p.device_id = device_id
                p.data = data_bytes
                self.tx_publisher.publish(p)
        

def main(args=None):
    rclpy.init(args=args)
    node = MasterArmControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

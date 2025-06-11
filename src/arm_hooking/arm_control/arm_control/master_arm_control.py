import rclpy
from rclpy.node import Node

from bpl_msgs.msg import Packet

# from bplprotocol import PacketReader
from rs_protocol import RSProtocol, create_socket_connection, create_serial_connection


class MasterArmControl(Node):
    def __init__(self):
        super().__init__('master_arm_control')

        # serial port for the Master Arm
        self.declare_parameter('master_arm_serial_port', '/dev/ttyUSB1')
        master_arm_port = self.get_parameter('master_arm_serial_port').value
        self.rs_protocol = RSProtocol(
            create_serial_connection(master_arm_port)
        )
        self.get_logger().info(f"Opened serial port {master_arm_port} for Master Arm")

        # publish Master Arm data to BPL Passthrough
        self.tx_publisher = self.create_publisher(Packet, "tx", 10)
        self.frequency = 100  # Hz
        self.create_timer(1.0/self.frequency, self.read_serial)

    def read_serial(self):
        packets = self.rs_protocol.read_raw()

        for packet in packets:
            # self.get_logger().info(f"Packet: {packet}")
            device_id, packet_id, data_bytes, options = packet
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

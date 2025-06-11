"""
ra_passthrough.py

Originally based on Reach Robotics' serial_passthrough.py for BPL Protocol; adapted to RS Protocol
Used to connect to an arm and forward received ros messages
"""

import rclpy
from rclpy.node import Node

from bpl_msgs.msg import Packet

from rs_protocol import RSProtocol, PacketID, Mode, create_socket_connection, create_serial_connection, encode_floats, decode_floats

import array


class RAPassthrough(Node):

    def __init__(self):
        super().__init__('ra_passthrough')

        self.declare_parameter('connection_type', 'serial')
        self.declare_parameter('serial_port',     '/dev/ttyUSB0')
        self.declare_parameter('ip_address',      '192.168.2.2')
        self.declare_parameter('udp_port',        6789)

        connection_type = self.get_parameter('connection_type').value
        self.rs_protocol = None
        if connection_type == 'udp':
            ip_address      = self.get_parameter('ip_address').value
            udp_port        = self.get_parameter('udp_port').value
            self.rs_protocol = RSProtocol(
                create_socket_connection(), 
                (ip_address, udp_port)
            )
        elif connection_type == 'serial':
            serial_port     = self.get_parameter('serial_port').value
            self.rs_protocol = RSProtocol(
                create_serial_connection(serial_port)
            )
        else:
            self.get_logger().error(f"Invalid connection type: {connection_type}. Please specify either udp or serial.")
            self.destroy_node()
            return

        self.rx_publisher = self.create_publisher(Packet,'rx', 5)
        self.tx_subscriber = self.create_subscription(Packet, "tx", self.tx_transmit, 5)

        self.timer = self.create_timer(1/10000000, self.rx_receive)

    def tx_transmit(self, packet):
        # receive from topic and write to port
        device_id = packet.device_id
        packet_id = packet.packet_id
        data = packet.data

        # if not packet_id in PacketID.PacketType:
        #     self.get_logger().info(f"Invalid packet_id: {packet_id}")
        #     return

        if isinstance(data, array.array):
            data = list(data)

        if PacketID.PacketType.get(packet_id) == float:
            if isinstance(data, list):
                data = bytes(data)
            data = decode_floats(data)

        self.rs_protocol.write(device_id, packet_id, data)
        # self.rs_protocol.request(device_id, packet_id)  # use to request info; may be equivalent to self.rs_protocol.write(device_id, PacketID.REQUEST, packet_id)

    def rx_receive(self):
        # read from port and publish to topic
        # packets = self.rs_protocol.read()
        packets = self.rs_protocol.read_raw()
        for packet in packets:
            device_id = packet[0]
            packet_id = packet[1]
            data = packet[2]

            ros_packet = Packet()
            ros_packet.device_id = device_id
            ros_packet.packet_id = packet_id
            ros_packet.data = data
            self.rx_publisher.publish(ros_packet)


def main(args=None):
    rclpy.init(args=args)
    node = RAPassthrough()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
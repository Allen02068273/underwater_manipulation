import rclpy
from rclpy.node import Node

# from bplprotocol import BPLProtocol, PacketID
from bpl_msgs.msg import Packet
from rs_protocol import PacketID, decode_floats

from std_msgs.msg import Float32MultiArray

class CurrentPublisher(Node):

    def __init__(self):
        super().__init__("CurrentPublisher")

        self.declare_parameter("frequency", 20)
        self.frequency = self.get_parameter("frequency").value

        self.tx_publisher = self.create_publisher(Packet, "tx", 100)
        self.rx_subscriber = self.create_subscription(Packet, "rx", self.receive_packet, 100)
        self.pose_publisher = self.create_publisher(Float32MultiArray, "current", 10)
        self.joints = [0x01, 0x02, 0x03, 0x04, 0x05]  # arm joints in order from gripper to base
        self.currents = [0.0] * len(self.joints)  # one current for each joint

        self.timer_request = self.create_timer(1/self.frequency, self.request_current)
        self.timer_publish = self.create_timer(1/self.frequency, self.publish_current)

    def request_current(self):
        for joint in self.joints:
            p = Packet()
            p.device_id = joint
            p.packet_id = int(PacketID.REQUEST)
            p.data = [PacketID.CURRENT]
            self.tx_publisher.publish(p)

    def receive_packet(self, packet):
        device_id = packet.device_id
        packet_id = packet.packet_id
        data = bytearray(packet.data)

        if packet_id == PacketID.CURRENT:
            current = decode_floats(data)

            if device_id in self.joints:
                index = self.joints.index(device_id)
                self.currents[index] = float(current[0])
    
    def publish_current(self):
        f = Float32MultiArray()
        f.data = self.currents
        self.pose_publisher.publish(f)
       

def main(args = None):
    rclpy.init(args=args)
    node = CurrentPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
import rclpy
from rclpy.node import Node

# from bplprotocol import BPLProtocol, PacketID
from bpl_msgs.msg import Packet
from rs_protocol import PacketID, decode_floats

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

from scipy.spatial.transform import Rotation

class EndEffectorPosePublisher(Node):

    def __init__(self):
        super().__init__("EndEffectorPosePublisher")

        self.declare_parameter("frequency", 20)
        self.declare_parameter("frame_id", "base_link")
        self.frame_id = self.get_parameter("frame_id").value
        self.frequency = self.get_parameter("frequency").value

        self.tx_publisher = self.create_publisher(Packet, "tx", 100)
        self.rx_subscriber = self.create_subscription(Packet, "rx", self.receive_packet, 100)
        self.pose_publisher = self.create_publisher(PoseStamped, "end_effector_pose", 10)
        self.request_packet = Packet()
        self.request_packet.device_id = 0xFF
        self.request_packet.packet_id = int(PacketID.REQUEST)
        self.request_packet.data = [PacketID.INVERSE_KINEMATICS_GLOBAL_POSITION]

        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

    def timer_callback(self):
        self.tx_publisher.publish(self.request_packet)

    def receive_packet(self, packet):
        device_id = packet.device_id
        packet_id = packet.packet_id
        data = bytearray(packet.data)

        if packet_id == PacketID.INVERSE_KINEMATICS_GLOBAL_POSITION:
            end_pos = decode_floats(data)
        
            self.publish_pose(end_pos)
    
    def publish_pose(self, end_pos):
        p = PoseStamped()

        p.pose.position.x = float(end_pos[0]*0.001)
        p.pose.position.y = float(end_pos[1]*0.001)
        p.pose.position.z = float(end_pos[2]*0.001)

        rot = Rotation.from_euler('xyz', end_pos[-3:][::-1]).as_quat()

        # Pose Orientation (Quaternion)
        p.pose.orientation.x = rot[0]
        p.pose.orientation.y = rot[1]
        p.pose.orientation.z = rot[2]
        p.pose.orientation.w = rot[3]

        p.header.frame_id = self.frame_id
        p.header.stamp = self.get_clock().now().to_msg()

        self.pose_publisher.publish(p)
       

def main(args = None):
    rclpy.init(args=args)
    node = EndEffectorPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, TwistStamped
from bpl_msgs.msg import Packet

# from bplprotocol import BPLProtocol, PacketID
from rs_protocol import PacketID, encode_floats

import math

from scipy.spatial.transform import Rotation as R

class ControlNode(Node):
    def __init__(self):
        super().__init__("bpl_control_node")

        self.declare_parameter('publish_frequency', 20)
        self.publish_frequency = self.get_parameter('publish_frequency').value
        
        self.tx_publisher = self.create_publisher(Packet, "tx", 10)

        self.joint_command_subscriber       = self.create_subscription(Float32MultiArray, "command/joint_positions", self.position_command_handler, 10)
        self.velocity_command_subscriber    = self.create_subscription(Float32MultiArray, "command/joint_velocities", self.velocity_command_handler, 10)
        self.km_command_subscriber          = self.create_subscription(PoseStamped, "command/km_command", self.km_command_handler, 10)
        self.ee_velocity_command_subscriber = self.create_subscription(TwistStamped, "command/ee_velocity", self.ee_velocity_command_handler, 10)

        self.joints = [0x01, 0x02, 0x03, 0x04, 0x05]  # arm joints in order from gripper to base
        self.position_command = None
        self.velocity_command = None
        self.km_command = None
        self.ee_velocity_command = None

        self.timer = self.create_timer(1/self.publish_frequency, self.publish_command)

    def position_command_handler(self, positions):
        self.position_command = positions.data

    def velocity_command_handler(self, velocities):
        self.velocity_command = velocities.data

    def km_command_handler(self, km_pos: PoseStamped):
        # m to mm
        x = km_pos.pose.position.x * 1000
        y = km_pos.pose.position.y * 1000
        z = km_pos.pose.position.z * 1000

        quat = km_pos.pose.orientation.x, km_pos.pose.orientation.y, km_pos.pose.orientation.z, km_pos.pose.orientation.w
        rot = R.from_quat(quat).as_euler('xyz')

        self.km_command = [x, y, z, rot[2], rot[1], rot[0]]

    def ee_velocity_command_handler(self, km_vel_stamped: TwistStamped):
        # m/s to mm/s
        x_dot = km_vel_stamped.twist.linear.x * 1000
        y_dot = km_vel_stamped.twist.linear.y * 1000
        z_dot = km_vel_stamped.twist.linear.z * 1000

        roll_dot = km_vel_stamped.twist.angular.x
        pitch_dot = km_vel_stamped.twist.angular.y
        yaw_dot = km_vel_stamped.twist.angular.z

        self.ee_velocity_command = [x_dot, y_dot, z_dot, yaw_dot, pitch_dot, roll_dot]

    def publish_command(self):
        if self.position_command is not None:
            for device_id, position in zip(self.joints, self.position_command):
                if math.isnan(position):
                    continue
                else:
                    p = Packet()
                    p.device_id = device_id
                    p.packet_id = PacketID.POSITION
                    p.data = list(encode_floats([position]))
                    self.tx_publisher.publish(p)

        if self.velocity_command is not None:
            for device_id, velocity in zip(self.joints, self.velocity_command):
                if math.isnan(velocity):
                    continue
                else:
                    p = Packet()
                    p.device_id = device_id
                    p.packet_id = PacketID.VELOCITY
                    p.data = list(encode_floats([velocity]))
                    self.tx_publisher.publish(p)

        if self.ee_velocity_command is not None:
            p = Packet()
            p.packet_id = PacketID.INVERSE_KINEMATICS_GLOBAL_VELOCITY
            p.device_id = 0x0E
            p.data = list(encode_floats(self.ee_velocity_command))
            self.tx_publisher.publish(p)

        if self.km_command is not None:
            p = Packet()
            p.packet_id = PacketID.INVERSE_KINEMATICS_GLOBAL_POSITION
            p.device_id = 0x0E
            p.data = list(encode_floats(self.km_command))
            self.tx_publisher.publish(p)


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
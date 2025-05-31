import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException, ConnectivityException
from geometry_msgs.msg import Pose
from tf2_geometry_msgs import do_transform_pose

import math
from scipy.spatial.transform import Rotation

from bplprotocol import BPLProtocol, PacketID
import serial

class InverseKinematics(Node):

    def __init__(self):
        super().__init__('inverse_kinematics')

        # parameter for serial port name to connect to the manipulator
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        serial_port_name = self.get_parameter('serial_port').value
        
        # transform listener and broadcaster for debugging
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # subscriber to desired end effector pose
        self.subscription_trajectory = self.create_subscription(PoseStamped, 'command/km_command', self.pose_callback, 10)

        # Reach Alpha 5 d and a DH parameters in m
        self.d = [0.0462, 0, 0, -0.180, 0]  # decrease d[3] to extend end point past gripper
        self.a = [0.020, math.sqrt(0.040**2 + 0.1453**2), 0.020, 0, 0]

        # shift 5cm to end effector instead of gripper
        # coeff = 1 - (0.050721818 / math.sqrt(self.d[3]**2 + self.a[2]**2))
        # self.d[3] *= coeff
        # self.a[2] *= coeff
        self.d[3] += 0.0492

        # serial port to send gripper position to arm (not ideal since everything else can work with serial or UDP)
        self.serial_port = serial.Serial(serial_port_name, baudrate=115200, parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE, timeout=0)

    def pose_callback(self, msg):
        pos = msg.pose.position
        x, y, z = pos.x, pos.y, pos.z #+ 0.0067
        mode = 'overarm pose'
        theta = [0,0,0]

        R = math.sqrt(x**2 + y**2)
        l1 = self.a[1]
        l2 = math.sqrt(self.a[2]**2 + self.d[3]**2)
        l3 = math.sqrt((R-self.a[0])**2 + (z-self.d[0])**2)
        if l3 > l1 + l2:
            l3 = l1 + l2
        theta[2] = math.acos((l1**2 + l2**2 - l3**2) / (2*l1*l2)) - math.asin((2*self.a[2]) / l1) - math.asin(self.a[2] / l2)
        if mode == 'underarm pose':
            theta[0] = math.atan2(y, x) + math.pi
            theta[1] = (math.pi / 2) + math.atan2(z-self.d[0], R-self.a[0]) - math.acos((l1**2 + l3**2 - l2**2) / (2*l1*l3)) - math.asin(2*self.a[2] / l1)
        else:
            theta[0] = math.atan2(y, x)
            theta[1] = (3*math.pi / 2) - math.atan2(z-self.d[0], R+self.a[0]) - math.acos((l1**2 + l3**2 - l2**2) / (2*l1*l3)) - math.asin(2*self.a[2] / l1)
        
        joint_angles = [3.35, 2.10, theta[2], theta[1], theta[0]]
        packets = b''
        for index, position in enumerate(joint_angles):
            device_id = index + 1
            packets += BPLProtocol.encode_packet(device_id, PacketID.POSITION, BPLProtocol.encode_floats([position]))
        self.serial_port.write(packets)

        self.broadcast_static_tfs('reach_alpha_base', 'target_pose_base', q=Rotation.from_euler('xyz', [0,0,theta[0]]).as_quat())
        self.broadcast_static_tfs('target_pose_base', 'target_pose_joint_1', position=(0.2,0,0), q=Rotation.from_euler('xyz', [0,theta[1],0]).as_quat())
        self.broadcast_static_tfs('target_pose_joint_1', 'target_pose_joint_2', position=(0.2,0,0), q=Rotation.from_euler('xyz', [0,theta[2],0]).as_quat())
        self.broadcast_static_tfs('target_pose_joint_2', 'target_pose_joint_3', position=(0.2,0,0))
        # self.get_logger().info(f"Thetas: {theta}")

    def broadcast_static_tfs(self, frame, child_frame, position=(0,0,0), q=(0,0,0,1)):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self.tf_static_broadcaster.sendTransform(t)
            

def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


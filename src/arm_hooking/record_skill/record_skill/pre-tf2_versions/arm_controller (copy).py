import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from pynput import keyboard

from bplprotocol import BPLProtocol, PacketID
import serial

'''
Keyboard Controls
w/s - up/down
a/d - left/right
q/e - forward/backward
r   - set and log robot's position as target position (use if the robot stops moving due to the target position being out of range)
f   - toggle gripper open/closed
g   - move to ready position
'''

class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller')

        # parameter for serial port name to connect to the manipulator
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        serial_port_name = self.get_parameter('serial_port').value

        # Publisher for pose
        self.pose_publisher_ = self.create_publisher(PoseStamped, 'command/km_command', 10)

        # end effector position and velocity variables
        self.pose_stamped = PoseStamped()
        self.pose_stamped.header.frame_id = "alpha_base_link"
        self.velocity = Twist()

        # time step (publishing rate, used in physics)
        self.time_step = 0.1 # seconds

        # start listening to keyboard input
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

        # timer to publish velocity and update position at a fixed rate
        self.timer = self.create_timer(self.time_step, self.update_pose_and_publish)

        # subscriber and variables to initialize/reset pose to robot pose
        self.pose_initialized = False
        self.reset_pose = False
        self.create_subscription(PoseStamped, 'end_effector_pose', self.pose_callback, 10)
        self.get_logger().info("Waiting for initial pose...")

        # serial port to send gripper position to arm (not ideal since everything else can work with serial or UDP)
        self.serial_port = serial.Serial(serial_port_name, baudrate=115200, parity=serial.PARITY_NONE,
                                    stopbits=serial.STOPBITS_ONE, timeout=0)
        self.gripper_open = False

    def on_key_press(self, key):
        # Handle keypress for velocity control
        try:
            if key.char == 'r': # reset pose
                self.pose_initialized = False
            elif key.char == 'f': # toggle gripper open/closed
                self.gripper_open = not self.gripper_open
            elif key.char == 'g': # move to ready position
                self.pose_stamped.pose.position.x = 0.000
                self.pose_stamped.pose.position.y = 0.024
                self.pose_stamped.pose.position.z = 0.179
            else: # move end effector
                acceleration = 0.01
                self.velocity.linear.z += acceleration * ((key.char == 'w') - (key.char == 's'))
                self.velocity.linear.x += acceleration * ((key.char == 'd') - (key.char == 'a'))
                self.velocity.linear.y += acceleration * ((key.char == 'q') - (key.char == 'e'))
                # self.get_logger().info(f"Velocity: {self.velocity.linear}")
        except AttributeError:
            pass  # Handle special keys if necessary

    def pose_callback(self, msg):
        if self.pose_initialized and not self.reset_pose:
            return
        
        # initialize the target pose as the robot's current pose
        self.pose_stamped.pose = msg.pose

        self.pose_initialized = True
        self.get_logger().info(f"Pose initialized at xyz {msg.pose.position.x} {msg.pose.position.y} {msg.pose.position.z}")

    def update_pose_and_publish(self):
        if not self.pose_initialized:
            return
        
        # update position based on velocity
        self.pose_stamped.pose.position.x += self.velocity.linear.x * self.time_step
        self.pose_stamped.pose.position.y += self.velocity.linear.y * self.time_step
        self.pose_stamped.pose.position.z += self.velocity.linear.z * self.time_step

        if self.pose_stamped.pose.position.y > 0.185: # arm will turn upside down if y goes beyond this value
            self.pose_stamped.pose.position.y = 0.185

        # dampen velocity
        dampen = 0.5
        self.velocity.linear.x *= dampen
        self.velocity.linear.y *= dampen
        self.velocity.linear.z *= dampen
        
        # timestamp and publish pose
        self.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        self.pose_publisher_.publish(self.pose_stamped)

        # send command to gripper
        if self.gripper_open:
            gripper_pos = [3.7, 2.10]
        else:
            gripper_pos = [3.35, 2.10]
        packets = b''
        for index, position in enumerate(gripper_pos):
            device_id = index + 1
            packets += BPLProtocol.encode_packet(device_id, PacketID.POSITION, BPLProtocol.encode_floats([position]))
        self.serial_port.write(packets)

def main(args=None):
    rclpy.init(args=args)
    node = ArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()

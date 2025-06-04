import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32MultiArray
from pynput import keyboard
import pygame
import math

'''
Keyboard Controls
 w/s - up/down
 a/d - left/right
 q/e - forward/backward
 r   - set and log robot's position as target position (use if the robot stops moving due to the target position being out of range)
 f   - toggle gripper open/closed
 g   - move to ready position

PS5 Controls
 left stick  - horizontal movement
 right stick - vertical movement
 circle      - set and log robot's position as target position (use if the robot stops moving due to the target position being out of range)
 cross       - toggle gripper open/closed
 triangle    - move to ready position
'''

class GamepadKeyboardControl(Node):
    def __init__(self):
        super().__init__('gamepad_keyboard_control')

        # Publisher for pose
        self.ee_publisher = self.create_publisher(PoseStamped, 'command/km_command', 10)
        self.gripper_publisher = self.create_publisher(Float32MultiArray, 'command/joint_positions', 10)

        # end effector position and velocity variables
        self.pose_stamped = PoseStamped()
        self.pose_stamped.header.frame_id = "alpha_base_link"
        self.velocity = Twist()

        # time step (publishing rate, used in physics)
        self.time_step = 0.1 # seconds

        # start listening to keyboard input
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Using controller: {self.joystick.get_name()}")
            self.timer = self.create_timer(0.03, self.handle_gamepad_input)
        else:
            print("No joystick detected!")

        # timer to publish velocity and update position at a fixed rate
        self.timer = self.create_timer(self.time_step, self.update_pose_and_publish)

        # subscriber and variables to initialize/reset pose to robot pose
        self.pose_initialized = False
        self.reset_pose = False
        self.create_subscription(PoseStamped, 'end_effector_pose', self.pose_callback, 10)
        self.get_logger().info("Waiting for initial pose...")
        
        self.gripper_open = False
        self.gripper_input_block = False

    def on_key_press(self, key):
        # Handle keypress for velocity control
        try:
            if key.char == 'r': # reset pose
                self.pose_initialized = False
            elif key.char == 'f': # toggle gripper open/closed
                self.gripper_open = not self.gripper_open
            elif key.char == 'g': # move to ready position
                self.pose_stamped.pose.position.x = 0.000
                self.pose_stamped.pose.position.y = -0.024
                self.pose_stamped.pose.position.z = 0.179
            else: # move end effector
                acceleration = 0.02

                # vertical motion
                self.velocity.linear.z += acceleration * ((key.char == 'w') - (key.char == 's'))

                # rotation about Z-axis (left/right rotation)
                rotation_speed = acceleration * ((key.char == 'd') - (key.char == 'a'))
                yaw = math.atan2(self.pose_stamped.pose.position.y, self.pose_stamped.pose.position.x)
                self.velocity.linear.x += rotation_speed * math.sin(yaw)
                self.velocity.linear.y += rotation_speed * -math.cos(yaw)

                # forward/backward motion in the current direction
                forward_speed = acceleration * ((key.char == 'q') - (key.char == 'e'))
                self.velocity.linear.x += forward_speed * math.cos(yaw)
                self.velocity.linear.y += forward_speed * math.sin(yaw)
                
                # self.get_logger().info(f"Velocity: {self.velocity.linear}")
        except AttributeError:
            pass  # Handle special keys if necessary

    def handle_gamepad_input(self):
        pygame.event.pump()  # Process events

        acceleration = 0.005
        input_right = self.joystick.get_axis(0)  # left stick vertical
        input_forward = -self.joystick.get_axis(1)  # left stick horizontal
        input_up = -self.joystick.get_axis(4)  # right stick vertical

        # account for stick drift
        min_value = 0.02  # increase if there is drift; decrease for more sensitivity
        input_right *= abs(input_right) > min_value
        input_forward *= abs(input_forward) > min_value
        input_up *= abs(input_up) > min_value

        # vertical motion
        self.velocity.linear.z += acceleration * input_up

        # rotation about Z-axis (left/right rotation)
        rotation_speed = acceleration * input_right
        yaw = math.atan2(self.pose_stamped.pose.position.y, self.pose_stamped.pose.position.x)
        self.velocity.linear.x += rotation_speed * math.sin(yaw)
        self.velocity.linear.y += rotation_speed * -math.cos(yaw)

        # forward/backward motion in the current direction
        forward_speed = acceleration * input_forward
        self.velocity.linear.x += forward_speed * math.cos(yaw)
        self.velocity.linear.y += forward_speed * math.sin(yaw)

        # Button handling
        if self.joystick.get_button(0):
            if self.gripper_input_block == False:  # cross button (X) -> toggle gripper
                self.gripper_open = not self.gripper_open
                self.gripper_input_block = True
        else:
            self.gripper_input_block = False
        if self.joystick.get_button(1):  # circle button (O) -> reset pose
            self.pose_initialized = False
        if self.joystick.get_button(2):  # triangle button -> move to ready position
            self.pose_stamped.pose.position.x = 0.000
            self.pose_stamped.pose.position.y = -0.024
            self.pose_stamped.pose.position.z = 0.179

    def pose_callback(self, msg):
        if self.pose_initialized and not self.reset_pose:
            return
        
        # avoid logging when end effector position is at origin (bug check)
        if msg.pose.position.x == 0.0 and msg.pose.position.y == 0.0 and msg.pose.position.z == 0.0:
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
        self.ee_publisher.publish(self.pose_stamped)

        # send command to gripper
        gripper_msg = Float32MultiArray()
        gripper_msg.data = [4.0, 2.10] if self.gripper_open else [2.0, 2.10]
        self.gripper_publisher.publish(gripper_msg)

def main(args=None):
    rclpy.init(args=args)
    node = GamepadKeyboardControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

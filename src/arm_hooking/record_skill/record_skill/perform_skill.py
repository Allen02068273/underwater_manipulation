import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Point, TransformStamped
from std_msgs.msg import Float32MultiArray
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException, ConnectivityException
from tf2_geometry_msgs import do_transform_pose

import numpy as np
from scipy.spatial.transform import Rotation as R
import control

import csv

class SkillPerformer(Node):

    def __init__(self):
        super().__init__('skill_performer')

        # parameter for the speed with which to follow the trajectory in m/s
        self.declare_parameter('speed', 0.05)
        self.speed = float(self.get_parameter('speed').value)
        # parameter for the tf2 frame of the target
        self.declare_parameter('target_frame', 'target_2')
        self.target_name = self.get_parameter('target_frame').value
        # parameter for whether to record the debug data
        self.declare_parameter('record_debug', True)
        self.record_debug = self.get_parameter('record_debug').value

        # transform listener for target/base/tool transforms
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # publisher for trajectory request
        self.request_publisher = self.create_publisher(Point, 'request_trajectory', 10)
        # subscriber to trajectory response
        self.subscription_trajectory = self.create_subscription(Float32MultiArray, 'gen_trajectory', self.trajectory_callback, 10)
        # publishers for end effector pose
        self.pose_publisher = self.create_publisher(PoseStamped, 'command/km_command', 10)
        self.gripper_publisher = self.create_publisher(Float32MultiArray, 'command/joint_positions', 10)
        # timers for running scripts till completion
        self.request_timer = self.create_timer(1.0, self.send_initial_position)
        self.time_step = 0.1
        self.step_timer = None

        # main transform for target frame to robot frame
        self.transform = None

        # trajectory following
        self.trajectory = None
        self.trajectory_index = None
        self.position = None  # position sent to robot in robot base frame
        self.virtual_target = None  # target moving along trajectory in target frame

        # tf broadcaster to add debugging tfs to tree
        self.tfb = TransformBroadcaster(self)

        # LQR parameters
        self.A = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [np.zeros((3, 3)), np.zeros((3, 3))]
        ])
        self.B = np.block([
            [np.zeros((3, 3))],
            [np.eye(3)]
        ])
        Q = np.diag([1, 1, 1, 1.25, 1.25, 1.25])  # penalize position error less than velocity to better preserve trajectory shape
        R = np.eye(3) * 0.1                       # penalize large accelerations (inputs)
        self.K, S, E = control.lqr(self.A, self.B, Q, R)

        # debugging CSV files
        self.csv_files = {}
        self.csv_writers = {}

        if self.record_debug:
            csv_filenames = ["virtual_target", "lqr_output", "robot_performance"]
            for name in csv_filenames:
                file_path = f"data/debug_trajectories/{name}.csv"
                f = open(file_path, mode='w', newline='')
                self.csv_files[name] = f
                writer = csv.writer(f)
                self.csv_writers[name] = writer
                writer.writerow(['timestamp', 'x', 'y', 'z'])
        
    def try_get_tf(self, frame, child_frame):
        # try getting the transform of one frame (child_frame) in another (frame)
        try:
            return self._tf_buffer.lookup_transform(frame, child_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException) as e:
            self.get_logger().info(f"Waiting for transform {repr(e)}.")
            return None
    
    # request a trajectory based on initial position
    def send_initial_position(self):
        # check for subscribers
        if self.request_publisher.get_subscription_count() == 0:
            self.get_logger().info("Waiting for subscriber to 'request_trajectory'.")
            return
        
        # try getting the transforms of the tool in the target frame and the 
        trans = self.try_get_tf(self.target_name, 'reach_alpha_tool')
        trans_init_pos = self.try_get_tf('reach_alpha_base', 'reach_alpha_tool')
        if trans is None or trans_init_pos is None:
            return

        # convert to Point object and publish
        pos = Point()
        pos.x, pos.y, pos.z = float(trans.transform.translation.x), float(trans.transform.translation.y), float(trans.transform.translation.z)
        self.request_publisher.publish(pos)
        self.request_timer.cancel()
        self.get_logger().info(f"Requested trajectory with initial position {pos}.")

        # initialize current arm position and 0 velocity
        self.position = np.array([trans_init_pos.transform.translation.x,
                                  trans_init_pos.transform.translation.y,
                                  trans_init_pos.transform.translation.z,
                                  0.0,0.0,0.0])

    def trajectory_callback(self, msg):
        # ensure only one trajectory is processed
        if self.trajectory is not None:
            return
        
        self.get_logger().info("Received trajectory from generator.")
        
        # received data length must be a multiple of 3 for x, y, z
        if len(msg.data) % 3 != 0:
            self.get_logger().error("Trajectory data length is not a multiple of 3.")
            return
        
        # reshape the flattened data into an N x 3 array
        self.trajectory = np.array(msg.data).reshape(-1, 3)
        self.get_logger().info(f"Trajectory contains {len(self.trajectory)} points.")
        
        self.trajectory_index = 0
        self.virtual_target = np.hstack((self.trajectory[0], np.array([0.0,0.0,0.0])))  # initialize position and 0 velocity
        
        # start a timer to publish each trajectory step
        self.step_timer = self.create_timer(self.time_step, self.step_trajectory)

    def step_trajectory(self):
        # try getting the transform of the target in the arm base frame
        # trans = self.transform or self.try_get_tf('reach_alpha_base', self.target_name)  # use this line to prevent continuous localization
        trans = self.try_get_tf('reach_alpha_base', self.target_name)  # use this line for continuous localization
        self.transform = trans
        if trans is None:
            return
        
        self.update_virtual_target()

        self.apply_lqr(trans)

        if self.trajectory_index < len(self.trajectory)-1:
            self.publish_pose(self.position[0], self.position[1], self.position[2], gripper_open=False)
        else:
            self.publish_pose(0.000, 0.120, 0.150, gripper_open=True)

        #self.step_timer.cancel()

        # broadcast/record debugging data
        if self.trajectory_index < len(self.trajectory)-1:
            self.broadcast_debugging_tf(self.virtual_target, self.target_name, 'virtual_target')
            if self.record_debug:
                self.csv_writers["virtual_target"].writerow([self.get_clock().now().to_msg().sec, self.virtual_target[0], self.virtual_target[1], self.virtual_target[2]])
                # self.csv_writers["lqr_output"].writerow([self.get_clock().now().to_msg().sec, self.position[0], self.position[1], self.position[2]])

                tfs = self.try_get_tf(self.target_name, 'reach_alpha_base')
                if tfs:
                    pos = tfs.transform.translation
                    self.csv_writers["lqr_output"].writerow([self.get_clock().now().to_msg().sec, pos.x, pos.y, pos.z])

                tfs = self.try_get_tf(self.target_name, 'reach_alpha_tool')
                if tfs:
                    pose = Pose()
                    pose.position.x, pose.position.y, pose.position.z = self.position[:3]
                    pos = do_transform_pose(pose, tfs).position
                    self.csv_writers["robot_performance"].writerow([self.get_clock().now().to_msg().sec, pos.x, pos.y, pos.z])

            # self.broadcast_debugging_tf(self.position, 'reach_alpha_base', 'LQR_adjusted_pose')
    
    def update_virtual_target(self):
        target = self.trajectory[self.trajectory_index]
        diff = target - self.virtual_target[:3]

        # if close enough to the target, move to target and track the next point
        if np.linalg.norm(diff) < (self.speed * self.time_step):
            self.virtual_target = np.hstack((target, diff / self.time_step))
            self.trajectory_index += 1
        else:  # otherwise, move towards the current target
            self.virtual_target[3:] = self.speed * diff / np.linalg.norm(diff)
            self.virtual_target[:3] += self.virtual_target[3:] * self.time_step
        
        if self.trajectory_index >= len(self.trajectory):
            self.trajectory_index = len(self.trajectory) - 1
            return True
        return False
    
    def apply_lqr(self, transform):
        pos = Pose()
        pos.position.x, pos.position.y, pos.position.z = self.virtual_target[:3]
        vel = Pose()
        vel.position.x, vel.position.y, vel.position.z = self.virtual_target[3:]

        pos = do_transform_pose(pos, transform)

        # pure rotation tf for velocity rotation
        transform_rot = TransformStamped()
        transform_rot.header = transform.header
        transform_rot.child_frame_id = transform.child_frame_id
        transform_rot.transform.rotation = transform.transform.rotation
        transform_rot.transform.translation.x = 0.0
        transform_rot.transform.translation.y = 0.0
        transform_rot.transform.translation.z = 0.0

        vel = do_transform_pose(vel, transform_rot)

        target_position = np.array([pos.position.x, pos.position.y, pos.position.z, vel.position.x, vel.position.y, vel.position.z])

        error = self.position - target_position
        u = -self.K @ error

        x_dot = self.A @ self.position + self.B @ u
        self.position += x_dot * self.time_step
    
    def publish_pose(self, x, y, z, gripper_open):
        # yaw rotation matrix (point away from origin)
        yaw = np.array([x, y])
        yaw /= np.linalg.norm(yaw)
        R_yaw = np.array([
            [yaw[0], -yaw[1], 0],
            [yaw[1],  yaw[0], 0],
            [0, 0, 1]
        ])
        
        # try getting the transforms of the end effector in the base frame and the tool in the end effector frame
        trans = self.try_get_tf('reach_alpha_base', 'reach_alpha_end_effector')
        trans_tool = self.try_get_tf('reach_alpha_end_effector', 'reach_alpha_tool')
        if trans is None or trans_tool is None:
            return
        t = trans.transform.translation
        q = trans.transform.rotation
        q = [q.x, q.y, q.z, q.w]

        # pitch rotation matrix (use previous position)
        _, pitch, _ = R.from_quat(q).as_euler('xyz')
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # combined rotation matrix
        rotation_matrix = R_yaw @ R_pitch

        # translate hook position backward to end effector position
        tool_pos = trans_tool.transform.translation
        offset_vector = -np.array([tool_pos.x, tool_pos.y, tool_pos.z])  # hook offset from end effector in meters
        rotated_offset = rotation_matrix @ offset_vector
        x += rotated_offset[0]
        y += rotated_offset[1]
        z += rotated_offset[2]

        # timestamp and publish pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "alpha_base_link"
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        self.pose_publisher.publish(pose_stamped)
        self.get_logger().info(f"Published trajectory step {self.trajectory_index + 1} at xyz {pose_stamped.pose.position.x} {pose_stamped.pose.position.y} {pose_stamped.pose.position.z}.")

        # set gripper joint positions
        gripper_position = [4.0, 2.10] if gripper_open else [2.0, 2.10]

        # publish gripper command
        gripper_msg = Float32MultiArray()
        gripper_msg.data = gripper_position
        self.gripper_publisher.publish(gripper_msg)

    def broadcast_debugging_tf(self, pos_vel, frame, child_frame):
        tfs = TransformStamped()
        tfs.header.stamp = self.get_clock().now().to_msg()
        tfs.header.frame_id = frame
        tfs._child_frame_id = child_frame
        tfs.transform.translation.x = pos_vel[0]
        tfs.transform.translation.y = pos_vel[1]
        tfs.transform.translation.z = pos_vel[2]

        # use velocity direction as tf orientation
        velocity = pos_vel[3:]
        if np.linalg.norm(velocity) > 0.001: # ignore velocity under 1mm/s
            vel_norm = velocity / np.linalg.norm(velocity)
            r = R.align_vectors([vel_norm], [[1, 0, 0]])[0]  # find rotation that maps x-axis (1,0,0) to vel_norm
            quat = r.as_quat()
        else:
            quat = [0.0, 0.0, 0.0, 1.0]  # no velocity: identity quaternion

        tfs.transform.rotation.x = quat[0]
        tfs.transform.rotation.y = quat[1]
        tfs.transform.rotation.z = quat[2]
        tfs.transform.rotation.w = quat[3]
        self.tfb.sendTransform(tfs)

    def __del__(self):
        for file in self.csv_files:
            file.close()

def main(args=None):
    rclpy.init(args=args)
    node = SkillPerformer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

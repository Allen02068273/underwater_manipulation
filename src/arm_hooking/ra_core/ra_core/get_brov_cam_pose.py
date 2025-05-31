import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import LookupException, ConnectivityException
from tf2_geometry_msgs import do_transform_pose

import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.monte_carlo import systematic_resample
from geometry_msgs.msg import PoseStamped

class GetBrovCameraPose(Node):

    ''' Transform frames:
         reach_alpha_base
         reach_alpha_end_effector
         reach_alpha_camera
         apriltag_{tag_id}
         target_{tag_id}
    '''
    def __init__(self):
        super().__init__('get_brov_camera_pose')

        # Unscented Kalman Filter for each tag
        self.ukfs = {}

        # subscribers to various poses
        self.sub_end_effector_pose = self.create_subscription(PoseStamped, 'end_effector_pose', self.end_effector_pose_callback, 10)
        self.sub_apriltag_pose = self.create_subscription(TransformStamped, 'apriltag_pose', self.apriltag_pose_callback, 10)

        # transform broadcasters for dynamic target transform data (multiple made because they run in separate threads)
        self.end_effector_tfb = TransformBroadcaster(self)
        self.apriltag_tfb = TransformBroadcaster(self)
        self.brov_tfb = TransformBroadcaster(self)
        # transform listener for target/base/tool transforms
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # transform broadcaster for static transforms
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.static_tfs = []
        self.add_static_tfs('map', 'reach_alpha_base')  # anchors the tf tree in rviz
        self.add_static_tfs('reach_alpha_end_effector', 'reach_alpha_camera', position=(-0.026, 0, 0.033))  # camera offset from end effector
        self.broadcast_static_tfs()
        self.static_tf_broadcast_timer = self.create_timer(1.0, self.broadcast_static_tfs)

    def add_static_tfs(self, frame, child_frame, position=(0,0,0), q=(0,0,0,1)):
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
        self.static_tfs.append(t)

    # initialize UKF
    def new_ukf(self, init_x):
        def state_transition_function(x, dt):
            # x = [x_pos, y_pos, z_pos, q_x, q_y, q_z, q_w]
            # potentially incorporate robot IMU
            return x

        # Define the measurement function (from sensor data)
        def measurement_function(x):
            # Assume sensor gives position and quaternion (or euler angles)
            return x[:7]  # [x, y, z, q_x, q_y, q_z, q_w]
        
        points = MerweScaledSigmaPoints(7, alpha=0.1, beta=2., kappa=0.)  # Set up sigma points
        ukf = UnscentedKalmanFilter(dim_x=7, dim_z=7, dt=0.1, fx=state_transition_function, hx=measurement_function, points=points)
        ukf.x = init_x
        ukf.P = np.eye(7)  # initial uncertainty
        ukf.R = np.diag([5.0]*3 + [10.0]*4)  # [pos noise, orientation noise]
        return ukf
    
    def broadcast_static_tfs(self):
        for t in self.static_tfs:
            #t.header.stamp = self.get_clock().now().to_msg()
            self.tf_static_broadcaster.sendTransform(t)

    def end_effector_pose_callback(self, msg):
        # avoid logging when end effector position is at origin (bug check)
        if msg.pose.position.x == 0.0 and msg.pose.position.y == 0.0 and msg.pose.position.z == 0.0:
            return
        
        msg.header.frame_id = 'reach_alpha_base'
        self.end_effector_tfb.sendTransform(self.pose_to_tf(msg, 'reach_alpha_end_effector'))

    def apriltag_pose_callback(self, msg):
        if msg.header.frame_id == 'reach_alpha_camera':
            # put the TransformStamped into a PoseStamped
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x, p.pose.position.y, p.pose.position.z = msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z
            p.pose.orientation.x = msg.transform.rotation.x
            p.pose.orientation.y = msg.transform.rotation.y
            p.pose.orientation.z = msg.transform.rotation.z
            p.pose.orientation.w = msg.transform.rotation.w
            tag_frame = msg._child_frame_id

            # transform the pose into the base frame
            try:
                trans_camera_to_base = self._tf_buffer.lookup_transform('reach_alpha_base', msg.header.frame_id, rclpy.time.Time())
            except (LookupException, ConnectivityException) as e:
                self.get_logger().info(f"Waiting for transform {repr(e)}.")
                return
            p.pose = do_transform_pose(p.pose, trans_camera_to_base)
            p.header.frame_id = 'reach_alpha_base'

            # apply the UKF
            position = np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
            orientation = np.array([p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w])
            measurement = np.hstack((position, orientation))
            if tag_frame not in self.ukfs:
                self.ukfs[tag_frame] = self.new_ukf(measurement)
            else:
                self.ukfs[tag_frame].predict()
                self.ukfs[tag_frame].update(measurement)

            self.apriltag_tfb.sendTransform(self.ukf_to_tf(self.ukfs[tag_frame],
                                                           p.header.stamp,
                                                           p.header.frame_id,
                                                           tag_frame))
        else:
            # invert to be brov_camera pose in tag frame
            # extract translation and rotation from TransformStamped
            t = [msg.transform.translation.x,
                msg.transform.translation.y,
                msg.transform.translation.z]
            q = [msg.transform.rotation.x,
                msg.transform.rotation.y,
                msg.transform.rotation.z,
                msg.transform.rotation.w]

            # create rotation object and invert it
            rot = R.from_quat(q)
            rot_inv = rot.inv()

            # invert translation: -R^T * t
            t_inv = -rot_inv.apply(t)

            # convert back to quaternion
            q_inv = rot_inv.as_quat()

            # fill PoseStamped with the inverted transform (camera in tag frame)
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x, p.pose.position.y, p.pose.position.z = t_inv
            p.pose.orientation.x = q_inv[0]
            p.pose.orientation.y = q_inv[1]
            p.pose.orientation.z = q_inv[2]
            p.pose.orientation.w = q_inv[3]
             
            try:
                trans_camera_to_base = self.brov_tf_buffer.lookup_transform('reach_alpha_base', tag_frame, rclpy.time.Time())
            except (LookupException, ConnectivityException) as e:
                self.get_logger().info(f"Waiting for transform {repr(e)}.")
                return
            p.pose = do_transform_pose(p.pose, trans_camera_to_base)
            p.header.frame_id = 'reach_alpha_base'

            # apply the UKF
            position = np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
            orientation = np.array([p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w])
            measurement = np.hstack((position, orientation))
            if self.brov_ukf is None:
                self.brov_ukf = self.new_ukf(measurement)
            else:
                self.brov_ukf.predict()
                self.brov_ukf.update(measurement)

            self.brov_tfb.sendTransform(self.ukf_to_tf(self.brov_ukf,
                                                       p.header.stamp,
                                                       tag_frame,
                                                       p.header.frame_id))
        
    def pose_to_tf(self, pose_stamped, child_frame):
        # take a PoseStamped object and return a TransformStamped object
        tfs = TransformStamped()
        tfs.header.stamp = pose_stamped.header.stamp
        tfs.header.frame_id = pose_stamped.header.frame_id
        tfs._child_frame_id = child_frame
        tfs.transform.translation.x = pose_stamped.pose.position.x
        tfs.transform.translation.y = pose_stamped.pose.position.y
        tfs.transform.translation.z = pose_stamped.pose.position.z
        tfs.transform.rotation.x = pose_stamped.pose.orientation.x
        tfs.transform.rotation.y = pose_stamped.pose.orientation.y
        tfs.transform.rotation.z = pose_stamped.pose.orientation.z
        tfs.transform.rotation.w = pose_stamped.pose.orientation.w
        return tfs
        
    def ukf_to_tf(self, ukf, stamp, parent_frame, child_frame):
        # take an UnscentedKalmanFilter object and return a TransformStamped object
        tfs = TransformStamped()
        tfs.header.stamp = stamp
        tfs.header.frame_id = parent_frame
        tfs._child_frame_id = child_frame
        tfs.transform.translation.x = ukf.x[0]
        tfs.transform.translation.y = ukf.x[1]
        tfs.transform.translation.z = ukf.x[2]
        tfs.transform.rotation.x = ukf.x[3]
        tfs.transform.rotation.y = ukf.x[4]
        tfs.transform.rotation.z = ukf.x[5]
        tfs.transform.rotation.w = ukf.x[6]
        return tfs

    def __del__(self):
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = GetBrovCameraPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

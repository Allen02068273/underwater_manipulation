import copy
import logging
import numpy as np

from numpy.linalg import inv, norm
from scipy.stats import multivariate_normal
from typing import Tuple

# KMP class by Lorenzo Busellato (lbusellato on GitHub)
# found at https://github.com/lbusellato/KMP_demos/blob/main/kmp/model/KMP.py

REALMIN = np.finfo(np.float64).tiny  # To avoid division by 0

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
)

class KMP:
    """Trajectory imitation and adaptation using Kernelized Movement Primitives.

    Parameters
    ----------
    l : int, default=0.5
        Lambda regularization factor for the minimization problem.
    alpha : float, default=40
        Coefficient for the covariance prediction.
    sigma_f : float, default=1
        Kernel coefficient.
    tol : float, default=0.0005
        Tolerance for the discrimination of conflicting points.
    time_driven_kernel : bool, default=True
        Enable/disable the use of the time-driven kernel. Realistically, should be turned off
        only in the position-only input demo.
    verbose : bool, default=True
        Enable/disable verbose output.
    """

    def __init__(
        self,
        l: float = 0.5,
        alpha: float = 40,
        sigma_f: float = 1.0,
        adaptation_tol: float = 0.0005,
        time_driven_kernel: bool = True,
        verbose: bool = False,
    ) -> None:
        self.l = l
        self.alpha = alpha
        self.sigma_f = sigma_f
        self.kl_divergence = None
        self.tol = adaptation_tol
        self.time_driven_kernel = time_driven_kernel
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(level=logging.DEBUG if verbose else logging.INFO)

    def set_waypoint(self, 
                     s: np.ndarray, 
                     xi: np.ndarray, 
                     sigma: np.ndarray) -> None:
        """Adds a waypoint to the database, checking for conflicts.

        Parameters
        ----------
        s : np.ndarray of shape (n_input_features,n_samples)
            Array of input vectors.
        xi : np.ndarray of shape (n_output_features,n_samples)
            Array of output vectors
        sigma : np.ndarray of shape (n_output_features,n_output_features,n_samples)
            Array of covariance matrices
        """
        for j in range(len(s)):
            # Loop over the reference database to find any conflicts
            min_dist = np.inf
            for i in range(self.N):
                dist = np.linalg.norm(self.s[:,i]-s[j])
                if  dist < min_dist:
                    min_dist = dist
                    id = i
            if min_dist < self.tol:
                # Replace the conflicting point
                self.s[:,id] = s[j]
                self.xi[:,id] = xi[j]
                self.sigma[:,:,id] = sigma[j]
            else:
                # Add the new point to the database
                self.s = np.concatenate((self.s, np.array(s[j]).reshape(1,-1)),axis=1)
                self.xi = np.concatenate((self.xi, xi[j].T),axis=1)
                self.sigma = np.concatenate((self.sigma, np.expand_dims(sigma[j],2)),axis=2)
        # Refit the model with the new data
        self.fit(self.s, self.xi, self.sigma)

    def __kernel_matrix(self, t1: float, t2: float) -> np.ndarray:
        """Computes the kernel matrix for the given input pair.

        Parameters
        ----------
        t1 : float
            The first input.
        t2 : float
            The second input.

        Returns
        -------
        kernel : np.ndarray of shape (n_features,n_features)
            The kernel matrix evaluated in the provided input pair.
        """

        def kernel(s1, s2):
            return np.exp(-self.sigma_f * norm(s1 - s2) ** 2)

        # Compute the kernels
        dt = 0.001
        ktt = kernel(t1, t2)
        if not self.time_driven_kernel:
            return ktt*np.eye(self.O)
        ktdt_tmp = kernel(t1, t2 + dt)
        kdtt_tmp = kernel(t1 + dt, t2)
        kdtdt_tmp = kernel(t1 + dt, t2 + dt)
        # Components of the matrix
        ktdt = (ktdt_tmp - ktt) / dt
        kdtt = (kdtt_tmp - ktt) / dt
        kdtdt = (kdtdt_tmp - ktdt_tmp - kdtt_tmp + ktt) / dt**2
        # Fill the kernel matrix
        O = self.O // 2
        kernel_matrix = np.block([[ktt*np.eye(O), ktdt*np.eye(O)],[kdtt*np.eye(O), kdtdt*np.eye(O)]])

        return kernel_matrix

    def fit(self, X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> None:
        """Train the model by computing the estimator matrix inv(K+lambda*sigma).

        Parameters
        ----------
        X : np.ndarray of shape (n_input_features,n_samples)
            Array of input vectors.
        mu : np.ndarray of shape (n_output_features,n_samples)
            Array of output vectors
        var : np.ndarray of shape (n_output_features,n_output_features,n_samples)
            Array of covariance matrices
        """
        self.s = copy.deepcopy(X)
        self.xi = copy.deepcopy(mu)
        self.sigma = copy.deepcopy(var)
        self.O, self.N = self.xi.shape
        k = np.zeros((self.N * self.O, self.N * self.O))
        # Construct the estimator
        for i in range(self.N):
            for j in range(self.N):
                kernel = self.__kernel_matrix(self.s[:, i], self.s[:, j])
                k[i * self.O : (i + 1) * self.O, j * self.O : (j + 1) * self.O] = kernel
                if i == j:
                    # Add the regularization terms on the diagonal
                    k[j * self.O : (j + 1) * self.O, i * self.O : (i + 1) * self.O] += (
                        self.l * self.sigma[:, :, i]
                    )
        self._estimator = inv(k)
        self._logger.info("KMP fit done.")

    def predict(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Carry out a prediction on the mean and covariance associated to the given input.

        Parameters
        ----------
        s : np.ndarray of shape (n_features,n_samples)
            The set of inputs to make a prediction of.

        Returns
        -------
        xi : np.ndarray of shape (n_features,n_samples)
            The array of predicted means.

        sigma : np.ndarray of shape (n_features,n_features,n_samples)
            The array of predicted covariance matrices.
        """
        xi = np.zeros((self.O, s.shape[1]))
        sigma = np.zeros((self.O, self.O, s.shape[1]))
        for j in range(s.shape[1]):
            k = np.zeros((self.O, self.N * self.O))
            Y = np.zeros(self.N * self.O)
            for i in range(self.N):
                k[:, i * self.O : (i + 1) * self.O] = self.__kernel_matrix(
                    s[:, j], self.s[:, i]
                )
                for h in range(self.O):
                    Y[i * self.O + h] = self.xi[h, i]
            xi[:, j] = k @ self._estimator @ Y
            sigma[:, :, j] = self.alpha * (
                self.__kernel_matrix(s[:, j], s[:, j]) - k @ self._estimator @ k.T
            )
        self._logger.info("KMP predict done.")
        #self.kl_divergence = self.KL_divergence(xi, sigma, self.xi, self.sigma)

        return xi, sigma

    def KL_divergence(self, xi, sigma, xi_ref, sigma_ref) -> float:
        kl_divs = []
        for i in range(self.N):
            # Create a multivariate distribution from data
            kmp_dist = multivariate_normal(xi[:, i], sigma[:, :, i])
            ref_dist = multivariate_normal(xi_ref[:, i], sigma_ref[:, :, i])
            # Evaluate the pdfs of the distributions
            kmp_pdf = kmp_dist.pdf(xi[:, i])
            ref_pdf = ref_dist.pdf(xi[:, i])
            # Compute the Kullback-Leibler Divergence
            kl_div = ref_pdf * np.log(ref_pdf / kmp_pdf)
            kl_divs.append(kl_div)
        # Normalize, since kl divs can range wildly from tiny to huge numbers apprently
        kl_divs = np.array(kl_divs)
        kl_divs /= norm(kl_divs)
        # Compute an aggregate value
        return np.mean(kl_divs)


import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from scipy import interpolate
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
#from kmp.model import KMP

class KMPTrajectoryNode(Node):
    def __init__(self):
        super().__init__('kmp_trajectory_generator')
        
        self.declare_parameter('trajectory_csv', 'data/trajectory.csv')
        csv_file = self.get_parameter('trajectory_csv').value
        
        self.traj = self.load_trajectory_from_csv(csv_file)
        sample_count = 60
        self.traj = self.smooth_trajectory(self.traj, sample_count)
        
        self.kmp = KMP(l=0.5, alpha=40, sigma_f=200, time_driven_kernel=False)
        self.fit_kmp(self.traj)

        # subscriber for trajectory requests
        self.subscription = self.create_subscription(
            Point,
            'request_trajectory',
            self.request_callback,
            10
        )

        # puisher for trajectory responses
        self.publisher = self.create_publisher(
            Float32MultiArray,
            'gen_trajectory',
            10
        )
        
        self.get_logger().info("KMP Trajectory Node Initialized.")

    def load_trajectory_from_csv(self, csv_file):
        """Load trajectory from CSV file and verify required columns."""
        try:
            df = pd.read_csv(csv_file)
            required_columns = {'x', 'y', 'z'}
            if not required_columns.issubset(df.columns):
                self.get_logger().error(f"CSV file missing required columns: {required_columns - set(df.columns)}. Shutting down.")
                rclpy.shutdown()
                return None
            
            traj = df[['x', 'y', 'z']].to_numpy()
            self.get_logger().info(f"Loaded trajectory from {csv_file}")
            return traj.T
        except Exception as e:
            self.get_logger().error(f"Failed to load CSV: {e}. Shutting down.")
            rclpy.shutdown()
            return None

    def smooth_trajectory(self, traj, sample_count):
        """Smooth trajectory using B-splines."""
        if traj.shape[0] < 2:
            self.get_logger().info("Trajectory must have at least 2 points for smoothing.")
            return traj

        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
        t = np.linspace(0, 1, len(traj))

        tck_x = interpolate.splrep(t, x, s=0)
        tck_y = interpolate.splrep(t, y, s=0)
        tck_z = interpolate.splrep(t, z, s=0)

        t_new = np.linspace(0, 1, sample_count)
        x_smooth = interpolate.splev(t_new, tck_x)
        y_smooth = interpolate.splev(t_new, tck_y)
        z_smooth = interpolate.splev(t_new, tck_z)

        return np.vstack((x_smooth, y_smooth, z_smooth)).T

    def fit_kmp(self, traj):
        """Train the KMP model on the given trajectory."""
        time_steps = np.linspace(0, 1, len(traj)).reshape(1, -1)
        mean_traj = traj.T  # Shape should be (features, time)
        cov_traj = np.tile(np.eye(3) * 1e-4, (1, 1, len(traj)))  # Small covariance

        self.kmp.fit(time_steps, mean_traj, cov_traj)
        self.get_logger().info("KMP model trained.")

    def request_callback(self, msg):
        """Generate and publish a new KMP trajectory when a request is received."""
        start_point = np.array([msg.x, msg.y, msg.z])
        end_point = self.traj[-1, :]  # Keep the original endpoint

        new_traj = self.apply_kmp(start_point, end_point)

        # Publish the generated trajectory
        traj_msg = Float32MultiArray()
        traj_msg.data = new_traj.flatten().tolist()
        self.publisher.publish(traj_msg)
        self.get_logger().info("Published new KMP trajectory.")

    def apply_kmp(self, start, end):
        """Generate a new trajectory using KMP with the given start and end points."""
        waypoints = [start, end]
        times = [0.0, 1.0]  # Start and end at normalized time range
        covariances = [np.eye(3) * 1e-4, np.eye(3) * 1e-4]

        for t, p, cov in zip(times, waypoints, covariances):
            self.kmp.set_waypoint([t], [p.reshape(1, -1)], [cov])

        # Generate new trajectory
        time_steps = np.linspace(0, 1, len(self.traj)).reshape(1, -1)
        mu_kmp, _ = self.kmp.predict(time_steps)

        return mu_kmp.T  # Shape (N, 3)

def main(args=None):
    rclpy.init(args=args)
    node = KMPTrajectoryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()






import numpy as np
import pandas as pd

csv_files = {'data/skill_data.csv'}

def load_trajectory_from_csv(self, csv_files):
    self.demos = []
    for csv_file in csv_files:
        required_columns = {'gripper_x', 'gripper_y', 'gripper_z'}
        df = pd.read_csv('skill_data.csv')
        if not required_columns.issubset(df.columns):
            self.get_logger().error(f"CSV file missing required columns: {required_columns - set(df.columns)}. Shutting down.")
            rclpy.shutdown()
            return None
        pos = df[['gripper_x', 'gripper_y', 'gripper_z']].values
        self.demos.append({'pos': pos})
    return np.concatenate([t['pos'] for t in self.demos], axis=1).T


def load_trajectory_from_csv(self, csv_file):
    """Load trajectory from CSV file and verify required columns."""
    try:
        df = pd.read_csv(csv_file)
        required_columns = {'gripper_x', 'gripper_y', 'gripper_z'}
        if not required_columns.issubset(df.columns):
            self.get_logger().error(f"CSV file missing required columns: {required_columns - set(df.columns)}. Shutting down.")
            rclpy.shutdown()
            return None
        
        traj = df[['gripper_x', 'gripper_y', 'gripper_z']].to_numpy()
        self.get_logger().info(f"Loaded trajectory from {csv_file}")
        return traj.T
    except Exception as e:
        self.get_logger().error(f"Failed to load CSV: {e}. Shutting down.")
        rclpy.shutdown()
        return None
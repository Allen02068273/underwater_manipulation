import numpy as np
import matplotlib.pyplot as plt
import control as ct

# 2D double integrator: state = [x, x_dot, y, y_dot]
A = np.array([[0, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 0]])
B = np.array([[0, 0],
              [1, 0],
              [0, 0],
              [0, 1]])
C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
D = np.zeros((2, 2))

# LQR weights
Q = np.diag([50, 1, 50, 1])
R = np.eye(2)

# LQR controller
K, _, _ = ct.lqr(A, B, Q, R)

# Kalman filter design
V = np.eye(4) * 0.01    # Process noise
W = np.eye(2) * 0.1     # Measurement noise
L, _, _ = ct.lqe(A, np.eye(4), C, V, W)

# Simulation parameters
dt = 0.02
T = 10
N = int(T / dt)
t = np.linspace(0, T, N)

# Trajectory: circle of radius 2
radius = 2
omega = 2 * np.pi / T
xref = radius * np.cos(omega * t)
yref = radius * np.sin(omega * t)
xref_dot = -radius * omega * np.sin(omega * t)
yref_dot =  radius * omega * np.cos(omega * t)
ref_traj = np.vstack([xref, xref_dot, yref, yref_dot])

# State initialization
x_lqg = np.zeros((4, N))
x_hat = np.zeros((4, N))
x_lqr = np.zeros((4, N))
y = np.zeros((2, N))
u_lqg = np.zeros((2, N))
u_lqr = np.zeros((2, N))

x_lqg[:, 0] = [0, 0, 0, 0]       # Start at origin
x_hat[:, 0] = [0, 0, 0, 0]       # Initial estimate
x_lqr[:, 0] = [0, 0, 0, 0]       # LQR starts at origin

np.random.seed(1)

for k in range(N-1):
    ref = ref_traj[:, k]
    # LQG: measurement with noise
    y[:, k] = C @ x_lqg[:, k] + np.random.multivariate_normal([0, 0], W)
    # LQG: control law (estimated state - reference)
    u_lqg[:, k] = -K @ (x_hat[:, k] - ref)
    # LQR: control law (true state - reference)
    u_lqr[:, k] = -K @ (x_lqr[:, k] - ref)
    # LQG: true system with process noise
    w = np.random.multivariate_normal([0, 0, 0, 0], V)
    x_lqg[:, k+1] = x_lqg[:, k] + (A @ x_lqg[:, k] + B @ u_lqg[:, k]) * dt + w * np.sqrt(dt)
    # LQR: true system with process noise (for fair comparison)
    w_lqr = np.random.multivariate_normal([0, 0, 0, 0], V)
    x_lqr[:, k+1] = x_lqr[:, k] + (A @ x_lqr[:, k] + B @ u_lqr[:, k]) * dt + w_lqr * np.sqrt(dt)
    # Kalman filter update (discrete-time approximation)
    x_hat_pred = x_hat[:, k] + (A @ x_hat[:, k] + B @ u_lqg[:, k]) * dt
    y_hat_pred = C @ x_hat_pred
    innovation = y[:, k] - y_hat_pred
    x_hat[:, k+1] = x_hat_pred + (L @ innovation) * dt

# Plot results
plt.figure(figsize=(8, 8))
plt.plot(xref, yref, 'r--', label='Reference trajectory')
plt.plot(x_lqg[0, :], x_lqg[2, :], 'b-', label='LQG (estimated state)')
plt.plot(x_lqr[0, :], x_lqr[2, :], 'g-', label='LQR (true state)')
plt.plot(x_hat[0, :], x_hat[2, :], 'k--', label='LQG state estimate')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Trajectory Tracking: LQG vs LQR')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()

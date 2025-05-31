import numpy as np
import control
import matplotlib.pyplot as plt

# System setup: 3D double integrator
# State: [x, y, z, vx, vy, vz]
# Dynamics: x_dot = vx, vx_dot = ax (acceleration as input)

A = np.block([
    [np.zeros((3, 3)), np.eye(3)],
    [np.zeros((3, 3)), np.zeros((3, 3))]
])

B = np.block([
    [np.zeros((3, 3))],
    [np.eye(3)]
])

# Define cost matrices
Q = np.diag([10, 10, 10, 1, 1, 1])  # Penalize position error more than velocity
R = np.eye(3) * 0.1                 # Penalize large accelerations (inputs)

# Solve for LQR gain
K, S, E = control.lqr(A, B, Q, R)

# Define initial and target states
start_pos = np.array([0.0, 0.0, 0.0])
start_vel = np.array([-0.05, -0.05, -0.05])
x = np.hstack((start_pos, start_vel))
target_pos = np.array([0.5, -0.5, 0.2])
target_vel = np.array([0.05, 0.05, 0.05])
x_target = np.hstack((target_pos, target_vel))

# Simulation parameters
dt = 0.01  # time step
T = 5.0    # total time
steps = int(T / dt)

# Thresholds
position_tolerance = 0.005  # meters

trajectory = []

# Simulation loop
for _ in range(steps):
    x_target[:3] += x_target[3:] * dt

    error = x - x_target
    u = -K @ error

    x_dot = A @ x + B @ u
    x += x_dot * dt

    trajectory.append(x.copy())

    position_error = np.linalg.norm(error[:3])
    print(f'position error: {position_error}')
    
    if position_error < position_tolerance:
        break

trajectory = np.array(trajectory)
target_pos = x_target[:3]

# Plot the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Trajectory')
ax.scatter(*start_pos, color='green', label='Start')
ax.scatter(*target_pos, color='red', label='Target')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('3D Point Moving with LQR')
plt.show()

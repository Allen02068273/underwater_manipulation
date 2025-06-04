import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("data/trajectory.csv")

# Create subplots
fig = plt.figure(figsize=(18, 6))

# Compute min/max values for each axis
x_min, x_max = df['x'].min(), df['x'].max()
y_min, y_max = (-df['y']).min(), (-df['y']).max()
z_min, z_max = df['z'].min(), df['z'].max()

# Compute min/max values for all data points
all_min = min(x_min, y_min, z_min, 0)
all_max = max(x_max, y_max, z_max, 0)

# Expand limits equally
padding = 0.05  # 5% padding
global_range = (all_max - all_min) * padding
all_min, all_max = all_min - global_range, all_max + global_range

# Plot back view
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(-df['y'].to_numpy(), df['z'].to_numpy(), label='Gripper Estimated Path', color='blue', marker='o', markersize=3, linestyle='-')
ax1.scatter(0, 0, color='red', marker='x', s=100, label="AprilTag")
ax1.set_xlabel("Y (Horizontal)")
ax1.set_ylabel("Z (Height)")
ax1.set_title("Back View")
ax1.set_xlim(all_min, all_max)
ax1.set_ylim(all_min, all_max)
ax1.legend()
ax1.grid()

# Plot side view
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(df['x'].to_numpy(), df['z'].to_numpy(), label='Gripper Estimated Path', color='blue', marker='o', markersize=3, linestyle='-')
ax2.scatter(0, 0, color='red', marker='x', s=100, label="AprilTag")
ax2.set_xlabel("X (Depth)")
ax2.set_ylabel("Z (Height)")
ax2.set_title("Side View")
ax2.set_xlim(all_min, all_max)
ax2.set_ylim(all_min, all_max)
ax2.legend()
ax2.grid()

# Plot top view
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(-df['y'].to_numpy(), df['x'].to_numpy(), label='Gripper Estimated Path', color='blue', marker='o', markersize=3, linestyle='-')
ax3.scatter(0, 0, color='red', marker='x', s=100, label="AprilTag")
ax3.set_xlabel("Y (Horizontal)")
ax3.set_ylabel("X (Depth)")
ax3.set_title("Top View")
ax3.set_xlim(all_min, all_max)
ax3.set_ylim(all_min, all_max)
ax3.legend()
ax3.grid()

plt.tight_layout()
plt.show()

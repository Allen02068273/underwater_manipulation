import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("data/trajectory_debug.csv")

# Create subplots
fig = plt.figure(figsize=(18, 6))

# Compute min/max values for each axis
x_min, x_max = df[['tool_x', 'target_x']].min().min(), df[['tool_x', 'target_x']].max().max()
y_min, y_max = df[['tool_y', 'target_y']].min().min(), df[['tool_y', 'target_y']].max().max()
z_min, z_max = df[['tool_z', 'target_z']].min().min(), df[['tool_z', 'target_z']].max().max()

# Compute min/max values for all data points
all_min = min(x_min, y_min, z_min)
all_max = max(x_max, y_max, z_max)

# Expand limits equally
padding = 0.05  # 5% padding
global_range = (all_max - all_min) * padding
all_min, all_max = all_min - global_range, all_max + global_range

# Plot back view (X vs Z)
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(df['tool_x'], df['tool_z'], label='Gripper Path', color='blue', marker='o', markersize=3, linestyle='-')
ax1.plot(df['target_x'], df['target_z'], label='Target Estimated Pose', color='orange', marker='x', markersize=3, linestyle='-')
ax1.set_xlabel("X (Horizontal)")
ax1.set_ylabel("Z (Height)")
ax1.set_title("Back View")
ax1.set_xlim(all_min, all_max)
ax1.set_ylim(all_min, all_max)
ax1.legend()
ax1.grid()

# Plot side view (Y vs Z)
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(df['tool_y'], df['tool_z'], label='Gripper Path', color='blue', marker='o', markersize=3, linestyle='-')
ax2.plot(df['target_y'], df['target_z'], label='Target Estimated Pose', color='orange', marker='x', markersize=3, linestyle='-')
ax2.set_xlabel("Y (Depth)")
ax2.set_ylabel("Z (Height)")
ax2.set_title("Side View")
ax2.set_xlim(all_min, all_max)
ax2.set_ylim(all_min, all_max)
ax2.legend()
ax2.grid()

# Plot top view (X vs Y)
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(df['tool_x'], df['tool_y'], label='Gripper Path', color='blue', marker='o', markersize=3, linestyle='-')
ax3.plot(df['target_x'], df['target_y'], label='Target Estimated Pose', color='orange', marker='x', markersize=3, linestyle='-')
ax3.set_xlabel("X (Horizontal)")
ax3.set_ylabel("Y (Depth)")
ax3.set_title("Top View")
ax3.set_xlim(all_min, all_max)
ax3.set_ylim(all_min, all_max)
ax3.legend()
ax3.grid()

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load CSV
df = pd.read_csv("data/trajectory.csv")

# Create figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Titles for each subplot
titles = ["Back View", "Side View", "Top View"]
axes_labels = [("Y (Horizontal)", "Z (Height)"), ("X (Depth)", "Z (Height)"), ("Z (Horizontal)", "X (Depth)")]

# Create empty plots
gripper_lines = []

for i, ax in enumerate(axes):
    ax.set_title(titles[i])
    ax.set_xlabel(axes_labels[i][0])
    ax.set_ylabel(axes_labels[i][1])
    ax.grid()
    ax.scatter(0, 0, color='red', marker='x', s=100, label="AprilTag")
    
    # Initialize empty line for gripper path
    (gripper_line,) = ax.plot([], [], 'bo-', marker='o', markersize=3, linestyle='-')
    
    gripper_lines.append(gripper_line)
    ax.legend()

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

def update(frame):
    for i, gripper_line in enumerate(gripper_lines):
        if i == 0:
            gripper_line.set_data(-df['gripper_y'][:frame], df['gripper_z'][:frame])
        elif i == 1:
            gripper_line.set_data(df['gripper_x'][:frame], df['gripper_z'][:frame])
        elif i == 2:
            gripper_line.set_data(-df['gripper_y'][:frame], df['gripper_x'][:frame])
        
        axes[i].set_xlim(all_min, all_max)
        axes[i].set_ylim(all_min, all_max)
        axes[i].set_aspect("equal")  # Ensures equal scaling

    return gripper_lines

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(df), interval=50, blit=False)

plt.show()

ani.save("data/trajectory_animation.mp4", writer="ffmpeg", fps=30)

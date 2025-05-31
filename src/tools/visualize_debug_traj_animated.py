import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load CSV
df = pd.read_csv("data/trajectory_debug.csv")

# Create figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Titles for each subplot
titles = ["Back View", "Side View", "Top View"]
axes_labels = [("X (Horizontal)", "Z (Height)"), ("Y (Depth)", "Z (Height)"), ("X (Horizontal)", "Y (Depth)")]

# Create empty plots
tool_lines = []
target_lines = []

for i, ax in enumerate(axes):
    ax.set_title(titles[i])
    ax.set_xlabel(axes_labels[i][0])
    ax.set_ylabel(axes_labels[i][1])
    ax.grid()
    
    # Initialize empty lines for gripper and target
    tool_line, = ax.plot([], [], 'bo-', markersize=3, label="Gripper Path")  # Blue line for gripper
    target_line, = ax.plot([], [], 'yx-', markersize=3, label="Target Estimated Pose")    # Orange line for target
    
    tool_lines.append(tool_line)
    target_lines.append(target_line)
    ax.legend()

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

def update(frame):
    for i, (tool_line, target_line) in enumerate(zip(tool_lines, target_lines)):
        if i == 0:  # X vs Z
            tool_line.set_data(df['tool_x'][:frame], df['tool_z'][:frame])
            target_line.set_data(df['target_x'][:frame], df['target_z'][:frame])
            axes[i].set_xlim(all_min, all_max)
            axes[i].set_ylim(all_min, all_max)
        elif i == 1:  # Y vs Z
            tool_line.set_data(df['tool_y'][:frame], df['tool_z'][:frame])
            target_line.set_data(df['target_y'][:frame], df['target_z'][:frame])
            axes[i].set_xlim(all_min, all_max)
            axes[i].set_ylim(all_min, all_max)
        elif i == 2:  # X vs Y
            tool_line.set_data(df['tool_x'][:frame], df['tool_y'][:frame])
            target_line.set_data(df['target_x'][:frame], df['target_y'][:frame])
            axes[i].set_xlim(all_min, all_max)
            axes[i].set_ylim(all_min, all_max)
        
        axes[i].set_aspect("equal")  # Ensures equal scaling

    return tool_lines + target_lines

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(df), interval=50, blit=True)

plt.show()

ani.save("data/trajectory_animation_debug.mp4", writer="ffmpeg", fps=30)

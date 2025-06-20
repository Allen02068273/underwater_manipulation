import pandas as pd
import matplotlib.pyplot as plt

def plot_xyz_from_csv(plot_title, csv_filename):
    """
    Reads a CSV file with columns 'x', 'y', and 'z', and plots each against time (row index).
    Y-axes are equalized so the scale is consistent across all subplots.
    """
    try:
        df = pd.read_csv(csv_filename)

        for col in ['x', 'y', 'z']:
            if col not in df.columns:
                raise ValueError(f"Missing column: '{col}'")

        time = df.index.to_numpy()
        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        z = df['z'].to_numpy()

        # Compute global min and max for equal y-axis scaling
        # y_min = min(x.min(), y.min(), z.min()) - 0.05
        # y_max = max(x.max(), y.max(), z.max()) + 0.05

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)
        fig.suptitle(plot_title, fontsize=16)

        axes[0].plot(time, x, color='r')
        axes[0].set_ylabel('X')
        # axes[0].set_ylim(y_min, y_max)
        axes[0].grid(True)

        axes[1].plot(time, y, color='g')
        axes[1].set_ylabel('Y')
        # axes[1].set_ylim(y_min, y_max)
        axes[1].grid(True)

        axes[2].plot(time, z, color='b')
        axes[2].set_ylabel('Z')
        axes[2].set_xlabel('Time (index)')
        # axes[2].set_ylim(y_min, y_max)
        axes[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)

    except Exception as e:
        print(f"Error: {e}")


plot_xyz_from_csv("Recorded Trajectory", "data/debug_trajectories/trajectory_.csv")
plot_xyz_from_csv("Recorded Trajectory 2", "data/debug_trajectories/trajectory.csv")
plot_xyz_from_csv("Generalized Trajectory", "data/debug_trajectories/generalized.csv")
plot_xyz_from_csv("Virtual Target", "data/debug_trajectories/virtual_target.csv")
plot_xyz_from_csv("Robot Performance", "data/debug_trajectories/robot_performance.csv")

plt.show()  # keeps the windows open

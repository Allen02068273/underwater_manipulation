import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Persistent figure and axes
_plot = {
    "fig": None,
    "axes": None,
}

def plot_xyz_overlay(label, csv_filename, color=None, plot_title=None):
    """
    Adds xyz plots from a CSV file to existing shared subplots.
    Normalizes the time axis so all plots fill the same horizontal range.

    Parameters:
        plot_title (str): Label for the dataset.
        csv_filename (str): CSV file with 'x', 'y', 'z' columns.
        label (str): Optional title of the figure (only used at first call).
        color (str): Optional color (e.g., 'r', 'g', 'b', etc.)
    """
    global _plot

    # Load and validate data
    df = pd.read_csv(csv_filename)
    for col in ['x', 'y', 'z']:
        if col not in df.columns:
            raise ValueError(f"Missing column: '{col}' in {csv_filename}")

    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()

    n = len(df)
    time = np.linspace(0, 1.0, n)  # normalize time

    # Create plot if not already initialized
    if not _plot["fig"]:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)
        fig.suptitle(plot_title, fontsize=16)
        _plot["fig"] = fig
        _plot["axes"] = axes

        for ax, dim in zip(axes, ['X', 'Y', 'Z']):
            ax.set_ylabel(dim)
            ax.grid(True)
        axes[-1].set_xlabel('Normalized Time')

    # Use current axes
    axes = _plot["axes"]

    # Plot on each subplot
    axes[0].plot(time, x, label=label, color=color)
    axes[1].plot(time, y, label=label, color=color)
    axes[2].plot(time, z, label=label, color=color)

    # Add legends if labels provided
    if label:
        for ax in axes:
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.draw()  # Redraw the updated plot without blocking


plot_xyz_overlay("Recorded", "data/debug_trajectories/raw.csv", plot_title="Trajectories")
# plot_xyz_overlay("Recorded 2", "data/debug_trajectories/trajectory.csv", plot_title="Trajectories")
plot_xyz_overlay("Smoothed", "data/debug_trajectories/smoothed.csv")
plot_xyz_overlay("Generalized", "data/debug_trajectories/generalized.csv")
plot_xyz_overlay("Virtual Target", "data/debug_trajectories/virtual_target.csv")
plot_xyz_overlay("LQR Output", "data/debug_trajectories/lqr_output.csv")
plot_xyz_overlay("Robot Performance", "data/debug_trajectories/robot_performance.csv")

plt.show()  # keeps the windows open

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# persistent plot data
_plot = {
    "fig": None,
    "axes": None,
    "lims": {"x": {"max": None, "min": None,},
             "y": {"max": None, "min": None,},
             "z": {"max": None, "min": None,},
    },
}

def plot_xyz_overlay(label, csv_filename, color=None, linestyle='-', marker=None, plot_title=None):
    """
    Adds xyz plots from a CSV file to existing shared subplots.
    Normalizes the time axis so all plots fill the same horizontal range.

    Parameters:
        plot_title (str): Label for the dataset.
        csv_filename (str): CSV file with 'x', 'y', 'z' columns.
        color (str): Optional color (e.g., 'r', 'g', 'b', etc.)
        linestyle (str): Optional line style (e.g., '-', '--', ':', etc.)
        plot_title (str): Optional title of the figure (only used at first call).
    """
    global _plot

    # load and validate data
    df = pd.read_csv(csv_filename)
    for col in ['x', 'y', 'z']:
        if col not in df.columns:
            raise ValueError(f"Missing column: '{col}' in {csv_filename}")

    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()

    n = len(df)
    time = np.linspace(0, 1.0, n)  # normalize time

    # create plot if not already initialized
    if not _plot["fig"]:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)
        fig.suptitle(plot_title, fontsize=16)
        _plot["fig"] = fig
        _plot["axes"] = axes

        for axis, data in zip(['x', 'y', 'z'], [x, y, z]):
            _plot["lims"][axis]["max"] = data.max()
            _plot["lims"][axis]["min"] = data.min()

        for ax, dim in zip(axes, ['X', 'Y', 'Z']):
            ax.set_ylabel(dim)
            ax.grid(True)
        axes[-1].set_xlabel('Normalized Time')

    # update limits and max range
    ranges = []
    for axis, data in zip(['x', 'y', 'z'], [x, y, z]):
        lims = _plot["lims"][axis]
        lims['max'] = max(data.max(), lims['max'])
        lims['min'] = min(data.min(), lims['min'])
        ranges.append(lims['max'] - lims['min'])
    plt_range = max(ranges)

    # use current axes
    axes = _plot["axes"]

    margin = 1.20  # at least 10% margin on top and bottom

    # set plot y limits
    for i, axis in enumerate(['x', 'y', 'z']):
        lims = _plot["lims"][axis]
        center = (lims['max'] + lims['min']) / 2
        plt_min = center - margin * plt_range / 2
        plt_max = center + margin * plt_range / 2
        axes[i].set_ylim(plt_min, plt_max)

    # plot on each subplot
    axes[0].plot(time, x, label=label, color=color, linestyle=linestyle, marker=marker)
    axes[1].plot(time, y, label=label, color=color, linestyle=linestyle, marker=marker)
    axes[2].plot(time, z, label=label, color=color, linestyle=linestyle, marker=marker)

    # add legends if labels provided
    if label:
        for ax in axes:
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.draw()  # redraw the updated plot without blocking


plot_xyz_overlay("Raw Demonstration", "data/debug_trajectories/raw.csv", linestyle=':', plot_title="Trajectories")
# plot_xyz_overlay("Recorded 2", "data/debug_trajectories/trajectory.csv", plot_title="Trajectories")
plot_xyz_overlay("Smoothed", "data/debug_trajectories/smoothed.csv")
plot_xyz_overlay("Generalized", "data/debug_trajectories/generalized.csv")
plot_xyz_overlay("LQR Target", "data/debug_trajectories/lqr_target.csv", linestyle='--')
plot_xyz_overlay("LQR Output", "data/debug_trajectories/lqr_output.csv")
plot_xyz_overlay("Robot Performance", "data/debug_trajectories/robot_performance.csv")

plt.show()  # keeps the windows open

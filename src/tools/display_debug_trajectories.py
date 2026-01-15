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

def plot_xyz_overlay(label, csv_filename, plot_via_point=False, color=None, linestyle='-', marker=None, plot_title=None):
    """
    Adds xyz plots from a CSV file to existing shared subplots.
    Normalizes the time axis so all plots fill the same horizontal range.

    Parameters:
        label (str): Label for the dataset.
        csv_filename (str): CSV file with 'x', 'y', 'z' columns.
        color (str): Optional color (e.g., 'r', 'g', 'b', etc.)
        linestyle (str): Optional line style (e.g., '-', '--', ':', etc.)
        marker (str): Optional marker style.
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
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 8), sharex=True)
        fig.suptitle(plot_title, fontsize=16)
        _plot["fig"] = fig
        _plot["axes"] = axes

        for axis, data in zip(['x', 'y', 'z'], [x, y, z]):
            _plot["lims"][axis]["max"] = data.max()
            _plot["lims"][axis]["min"] = data.min()

        for ax, dim in zip(axes, ['X', 'Y', 'Z']):
            ax.set_ylabel(dim, fontsize=14)
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=12)
        axes[-1].set_xlabel('Normalized Time', fontsize=14)

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
    axes[0].plot(time, x, label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2.5)
    axes[1].plot(time, y, label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2.5)
    axes[2].plot(time, z, label=label, color=color, linestyle=linestyle, marker=marker, linewidth=2.5)

    marker_color = 'black'

    # plot start point
    axes[0].plot(0, x[0], marker='o', linestyle='None', label='_start', color=marker_color)
    axes[1].plot(0, y[0], marker='o', linestyle='None', label='_start', color=marker_color)
    axes[2].plot(0, z[0], marker='o', linestyle='None', label='_start', color=marker_color)

    # plot end point
    axes[0].plot(1, x[-1], marker='s', linestyle='None', label='_end', color=marker_color)
    axes[1].plot(1, y[-1], marker='s', linestyle='None', label='_end', color=marker_color)
    axes[2].plot(1, z[-1], marker='s', linestyle='None', label='_end', color=marker_color)

    # plot via point
    if plot_via_point:
        via_index = int(.65 * len(x))
        via_time = via_index / (len(x)-1)
        axes[0].plot(via_time, x[via_index], marker='x', linestyle='None', label='_via', color=marker_color, markersize=10, markeredgewidth=2.5)
        axes[1].plot(via_time, y[via_index], marker='x', linestyle='None', label='_via', color=marker_color, markersize=10, markeredgewidth=2.5)
        axes[2].plot(via_time, z[via_index], marker='x', linestyle='None', label='_via', color=marker_color, markersize=10, markeredgewidth=2.5)

    # add legends if labels provided
    # if label:
    #     for ax in axes:
    #         ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.draw()  # redraw the updated plot without blocking

filepath = 'data/debug_trajectories/'
# plot_xyz_overlay("Raw Demonstration", f"{filepath}raw.csv", linestyle=':', plot_title="Trajectories")
# plot_xyz_overlay("Recorded 2", f"{filepath}trajectory.csv", plot_title="Trajectories")
plot_xyz_overlay("Preprocessed Demonstration", f"{filepath}smoothed.csv", linestyle='-.')
plot_xyz_overlay("Generalized Reproduction", f"{filepath}generalized.csv", plot_via_point=True, linestyle='--')
plot_xyz_overlay("LQR Target", f"{filepath}lqr_target.csv", linestyle='--')
plot_xyz_overlay("LQR Output", f"{filepath}lqr_output.csv")
plot_xyz_overlay("Robot Performance", f"{filepath}robot_performance.csv")

plt.show()  # keeps the windows open

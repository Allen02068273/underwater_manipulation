"""
Linear Quadratic Regulator (LQR) Class - Test

Test for the time-independent and time-varying versions of LQR

Code: Reza Azadeh - 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from lqr import LQR


def x_desired(t):
    pos = np.sin(t)
    vel = np.cos(t)
    return np.array([pos, vel])


def simulate_lqr(lqr, T, dt, label="LQR"):
    time = np.arange(0, T + dt, dt)
    n = lqr.A.shape[0]
    x = np.zeros((len(time), n))
    u = np.zeros(len(time))
    K_hist = np.zeros((len(time), n))  # Always save gains

    x[0] = np.zeros(n)

    # Initialize K_hist at the first step
    if lqr.mode == "time-invariant":
        K_hist[0] = lqr.K
    else:
        K_hist[0] = lqr.K_list[0]

    for i in range(len(time) - 1):
        t = time[i]
        x_d = x_desired(t)
        if lqr.mode == "time-invariant":
            u[i] = lqr.feedback(x[i], x_d)
            K_hist[i + 1] = lqr.K
        else:
            u[i] = lqr.feedback(x[i], x_d, t)
            K_hist[i + 1] = lqr.K_list[int(np.round(t / dt))]
        dx = lqr.A @ x[i] + lqr.B.flatten() * u[i]
        x[i + 1] = x[i] + dx * dt

    # Fill the last control input (optional)
    u[-1] = u[-2]

    x_d_all = np.array([x_desired(t) for t in time])

    # Make sure arrays have same length for plotting
    L = min(len(time), x.shape[0], u.shape[0], x_d_all.shape[0], K_hist.shape[0])
    time, x, u, x_d_all, K_hist = time[:L], x[:L], u[:L], x_d_all[:L], K_hist[:L]
    return time, x, u, x_d_all, K_hist


def plot_results(time, x, u, x_d_all, K_hist=None, title="LQR"):
    plt.figure(figsize=(10, 7))
    plt.subplot(3, 1, 1)
    plt.plot(time, x[:, 0], label="Actual position")
    plt.plot(time, x_d_all[:, 0], "--", label="Desired position")
    plt.ylabel("Position")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(time, x[:, 1], label="Actual velocity")
    plt.plot(time, x_d_all[:, 1], "--", label="Desired velocity")
    plt.ylabel("Velocity")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(time, u, label="Control input")
    if K_hist is not None:
        ax2 = plt.gca().twinx()
        ax2.plot(time, K_hist[:, 0], "r--", alpha=0.4, label="K[0]")
        ax2.plot(time, K_hist[:, 1], "g--", alpha=0.4, label="K[1]")
        ax2.set_ylabel("Gain value")
    plt.xlabel("Time [s]")
    plt.ylabel("Control")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def test_lqr_time_invariant():
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.diag([10, 1])
    R = np.diag([[0.1]]).reshape(1, 1)
    lqr = LQR(A, B, Q, R, mode="time-invariant")
    T = 10
    dt = 0.01
    time, x, u, x_d_all, K_hist = simulate_lqr(lqr, T, dt)
    plot_results(time, x, u, x_d_all, title=f"Time-Invariant LQR K={lqr.K}")


def test_lqr_time_varying_default_terminal():
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.diag([10, 1])
    R = np.diag([[0.1]]).reshape(1, 1)
    T = 10
    dt = 0.01
    lqr = LQR(A, B, Q, R, T=T, dt=dt, mode="time-varying")
    time, x, u, x_d_all, K_hist = simulate_lqr(lqr, T, dt)
    plot_results(
        time, x, u, x_d_all, K_hist, title=f"Time-Varying LQR (default terminal)"
    )


def test_lqr_time_varying_custom_terminal():
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    Q = np.diag([10, 1])
    R = np.diag([[0.1]]).reshape(1, 1)
    T = 10
    dt = 0.01
    terminal_penalty = np.diag([100, 10])
    lqr = LQR(
        A, B, Q, R, T=T, dt=dt, mode="time-varying", terminal_penalty=terminal_penalty
    )
    time, x, u, x_d_all, K_hist = simulate_lqr(lqr, T, dt)
    plot_results(
        time,
        x,
        u,
        x_d_all,
        K_hist,
        title=f"Time-Varying LQR (custom terminal: {terminal_penalty})",
    )


if __name__ == "__main__":
    test_lqr_time_invariant()
    test_lqr_time_varying_default_terminal()
    test_lqr_time_varying_custom_terminal()

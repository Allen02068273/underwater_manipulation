"""
Linear Quadratic Regulator (LQR) Class

This class implements both time-independent and time-varying versions of LQR

Code: Reza Azadeh - 2025
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are


class LQR:
    def __init__(
        self, A, B, Q, R, T=None, dt=None, mode="time-invariant", terminal_penalty=None
    ):
        """
        Parameters:
            A, B: System matrices
            Q, R: Cost matrices
            T: Final time (finite horizon for TV LQR).
            dt: time discretization (needed for TV LQR).
            mode: "time-invariant" or "time-varying"
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.T = T
        self.dt = dt
        self.mode = mode
        self.terminal_penalty = terminal_penalty if terminal_penalty is not None else Q
        self._compute_gains()

    def _riccati_ode(self, t, P_flat):
        n = self.A.shape[0]
        P = P_flat.reshape(n, n)
        dP = -(
            self.A.T @ P
            + P @ self.A
            - P @ self.B @ np.linalg.inv(self.R) @ self.B.T @ P
            + self.Q
        )
        return dP.flatten()

    def _compute_gains(self):
        if self.mode == "time-invariant":
            # Solve time-invariant ARE for infinite-horizon LQR
            P = solve_continuous_are(self.A, self.B, self.Q, self.R)
            self.K = np.linalg.inv(self.R) @ self.B.T @ P
            self.K = self.K.squeeze()
        elif self.mode == "time-varying":
            assert (
                self.T is not None and self.dt is not None
            ), "T and dt must be specified for time-varying LQR."
            n = self.A.shape[0]
            time = np.arange(0, self.T + self.dt, self.dt)
            # Choose terminal penalty for Riccati equation
            P_T = self.terminal_penalty.flatten()
            sol = solve_ivp(
                self._riccati_ode, [self.T, 0], P_T, t_eval=time[::-1], method="RK45"
            )
            Ps = sol.y.T[::-1]
            self.time = time
            self.K_list = [
                (np.linalg.inv(self.R) @ self.B.T @ P.reshape(n, n)).squeeze()
                for P in Ps
            ]
        else:
            raise ValueError("mode must be 'time-invariant' or 'time-varying'")

    def feedback(self, x, x_ref, t=None):
        """
        Compute control input for current state x and reference x_ref.

        For time-varying, t is used to index the gain schedule.
        """
        x_error = x - x_ref
        if self.mode == "time-invariant":
            u = -self.K @ x_error
        elif self.mode == "time-varying":
            assert t is not None, "Time t required for time-varying LQR."
            idx = int(np.round(t / self.dt))
            if idx >= len(self.K_list):
                idx = -1
            K = self.K_list[idx]
            u = -K @ x_error
        return u

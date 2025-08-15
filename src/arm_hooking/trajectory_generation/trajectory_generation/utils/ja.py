import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import solve_bvp

### JA (Jerk-Accuracy) Psuedocode
# Original: https://www.mathworks.com/matlabcentral/fileexchange/58403-kinematic-filtering-for-human-and-robot-trajectories

## Given: Trajectory X, Constraints C, Tradeoff Value Lambda
## Output: Deformed Trajectory X' which meets constraints
## Note: Lambda determines the tradeoff between Jerk and Accuracy. Lower values are lower jerk & accuracy, higher values are higher jerk & accuracy.

# 1. For each dimension,
# 2. -- Set up constraints (initial & endpoint position, velocity, acceleration)
# 3. -- Set up boundary value problem (BVP) (boundary conditions, differential equation, guess function)
# 4. -- Solve BVP
# 5. -- If using slow method, solve BVP again

#note: C is broken down into constraints and indices corresponding to those constraints in code

def JA_single(x, init, end, lmbda, method, time=None):
    n_pts = len(x)
    if time is None:
        time = np.linspace(0, 1, n_pts)
    
    #set up interpolated cubic spline over the time data to estimate function values
    F = InterpolatedUnivariateSpline(time, x, k=3)
    
    #function handles
    def bc(y0, y1):
        res = np.array([y0[0] - init[0], y0[1] - init[1], y0[2] - init[2], y1[0] - end[0], y1[1] - end[1], y1[2] - end[2]])#.reshape(6)
        return res
      
    #MSD is different from original due to how the solve_bvp function works - same mathematical operations 
    def MSDAccuracy_DE(t, y):
        p = 6 #default is 6
        dydt = np.zeros(np.shape(y))
        for i in range (len(t)):
            dydt[0, i] = y[1, i]
            dydt[1, i] = y[2, i]
            dydt[2, i] = y[3, i]
            dydt[3, i] = y[4, i]
            dydt[4, i] = y[5, i]
            dydt[5, i] = (lmbda**p) * (y[0, i] - F(t[i]))
        return dydt
      
    #create guess function
    t1 = time[0]
    t2 = time[-1]
    x0 = init[0] - F(t1)
    xf = end[0] - F(t2)
    
    #derivitaves guess
    denom = (t1 - t2)**5
    a0 = x0 + (t2 * (5 * (t1**4) * x0 - 5 * (t1**4) * xf) - (t1**5) * x0 + (t1**5) * xf - (t2**2) * (10 * (t1**3) * x0 - 10 * (t1**3) * xf)) / denom
    a1 = (30 * (t1**2) * (t2**2) * (x0-xf)) / denom
    a2 = -(30 * t1 * t2 * (t1 + t2) * (x0 - xf)) / denom
    a3 = (10 * (x0 - xf) * ((t1**2) + 4 * t1 * t2 + (t2**2))) / denom
    a4 = -(15 * (t1 + t2) * (x0 - xf)) / denom
    a5 = (6 * (x0 - xf)) / denom
    
    y = np.zeros((6, n_pts))
    for t in range (len(time)):
        y[0, t] = F(time[t]) + a5 * (time[t]**5) + a4 * (time[t]**4) + a3 * (time[t]**3) + a2 * (time[t]**2) + a1 * time[t] + a0
        y[1, t] = 5 * a5 * (time[t]**4) + 4 * a4 * (time[t]**3) + 3 * a3 * (time[t]**2) + 2 * a2 * time[t] + a1
        y[2, t] = 20 * a5 * (time[t]**3) + 12 * a4 * (time[t]**2) + 6 * a3 * time[t] + 2 * a2
        y[3, t] = 60 * a5 * (time[t]**2) + 24 * a4 * time[t] + 6 * a3
        y[4, t] = 120 * a5 * time[t] + 24 * a4
        y[5, t] = 120 * a5
      
    #solve the BVP 
    sol = solve_bvp(MSDAccuracy_DE, bc, time, y, max_nodes=n_pts)
    
    #Check if it didn't converge - would be due to bad guess
    if sol.status != 0:
        print("WARNING: sol.status is %d" % sol.status)
        print(sol.message)
      
    #slow indicates a recalculation of the BVP with the guess using the previously found solution
    if method == 'slow':
        sol = solve_bvp(MSDAccuracy_DE, bc, sol.x, sol.y, max_nodes=n_pts)
        if sol.status != 0:
            print("WARNING: sol.status is %d" % sol.status)
            print(sol.message)
    return sol.y[0]

def JA(X, C=None, inds=[], lmbda=None, time=None, C_vel=None, C_accel=None, method='fast'):
    import numpy as np, math
    
    def get_val(arr, i, d):
        """Extract arr[i][d] safely whether it's a scalar or an array-like."""
        val = arr[i][d]
        if np.isscalar(val):
            return val
        return val[0]  # works for (1,) arrays like in the demo

    n_pts, n_dims = np.shape(X)
    # Automatic guess for lambda
    if lmbda is None:
        lmbda = math.ceil(np.size(X) / 20.0)
    lmbda = lmbda * (n_pts / 250.0) * (n_dims**2 / 4.0)
    
    # Solve 1D bvp
    new_X = np.zeros((n_pts, n_dims))
    for d in range(n_dims):
        init = [X[0, d], 0, 0]
        end = [X[-1, d], 0, 0]
        for i in range(len(inds)):
            if inds[i] == 0:
                init[0] = get_val(C, i, d)
                if C_vel is not None:
                    init[1] = get_val(C_vel, i, d)
                if C_accel is not None:
                    init[2] = get_val(C_accel, i, d)
            if inds[i] == -1 or inds[i] == n_pts - 1:
                end[0] = get_val(C, i, d)
                if C_vel is not None:
                    end[1] = get_val(C_vel, i, d)
                if C_accel is not None:
                    end[2] = get_val(C_accel, i, d)
        traj1d = JA_single(X[:, d], init, end, lmbda, method, time)
        new_X[:, d] = traj1d
    return new_X

def main2D():
    # demonstration
    num_points = 50
    t = np.linspace(0, 10, num_points).reshape((num_points, 1))
    x_demo = np.sin(t) + 0.01 * t**2 - 0.05 * (t-5)**2
    y_demo = np.cos(t) - 0.01 * t - 0.03 * t**2
    print(x_demo.shape, y_demo.shape)

    traj = np.hstack((x_demo, y_demo))

    new_traj = JA(traj, np.array([[x_demo[0] , y_demo[0]],  [x_demo[-1], y_demo[-1]]]), [0, num_points-1], lmbda=85)
    
    new_traj2 = JA(traj, np.array([[x_demo[0]+0.5, y_demo[0]-0.2], [x_demo[-1], y_demo[-1]]]), [0, num_points-1], lmbda=85)
    
    new_traj3 = JA(traj, lmbda=65)
    
    plt.rcParams['figure.figsize'] = (6.5, 6.5)
    fig, axs = plt.subplots(2, 2)
    axs[0][0].set_title('Demonstration')
    axs[1][0].set_title('Same Constraints, lambda=85')
    axs[0][1].set_title('New Initial Point')
    axs[1][1].set_title('Same Constraints, lambda=65')
    axs[0][0].plot(traj[:, 0], traj[:, 1], 'k', lw=3)
    axs[1][0].plot(traj[:, 0], traj[:, 1], 'k', lw=3)
    axs[0][1].plot(traj[:, 0], traj[:, 1], 'k', lw=3)
    axs[1][1].plot(traj[:, 0], traj[:, 1], 'k', lw=3)
    
    axs[1][0].plot(new_traj[:, 0], new_traj[:, 1], 'm', lw=3)
    axs[1][0].plot(x_demo[0], y_demo[0], 'k.', ms=10)
    axs[1][0].plot(x_demo[-1], y_demo[-1], 'k.', ms=10)
    axs[1][0].plot(x_demo[0], y_demo[0], 'rx', ms=10, mew=2)
    axs[1][0].plot(x_demo[-1], y_demo[-1], 'rx', ms=10, mew=2)
    
    axs[0][1].plot(new_traj2[:, 0], new_traj2[:, 1], 'g', lw=3)
    axs[0][1].plot(x_demo[0], y_demo[0], 'k.', ms=10)
    axs[0][1].plot(x_demo[-1], y_demo[-1], 'k.', ms=10)
    axs[0][1].plot(x_demo[0]+0.5, y_demo[0]-0.2, 'rx', ms=10, mew=2)
    axs[0][1].plot(x_demo[-1], y_demo[-1], 'rx', ms=10, mew=2)
    
    axs[1][1].plot(new_traj3[:, 0], new_traj3[:, 1], 'b', lw=3)
    axs[1][1].plot(x_demo[0], y_demo[0], 'k.', ms=10)
    axs[1][1].plot(x_demo[-1], y_demo[-1], 'k.', ms=10)
    axs[1][1].plot(x_demo[0], y_demo[0], 'rx', ms=10, mew=2)
    axs[1][1].plot(x_demo[-1], y_demo[-1], 'rx', ms=10, mew=2)
    
    plt.show()
    
def main3D():
    # demonstration
    num_points = 50
    t = np.linspace(0, 10, num_points).reshape((num_points, 1))
    x_demo = np.sin(t) + 0.01 * t**2 - 0.05 * (t-5)**2
    y_demo = np.cos(t) - 0.01 * t - 0.03 * t**2
    z_demo = 2 * np.cos(t) * np.sin(t) - 0.01 * t**1.5 + 0.03 * t**2
    
    traj = np.hstack((x_demo, y_demo, z_demo))

    new_traj = JA(traj, [np.array([x_demo[0] + 0.5, y_demo[0] - 0.5, z_demo[0] + 0.5]), np.array([x_demo[-1], y_demo[-1], z_demo[-1]])], [0, num_points-1], lmbda=50)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot(x_demo, y_demo, z_demo, 'k', lw=3)
    ax.plot(new_traj[:, 0], new_traj[:, 1], new_traj[:, 2], 'm', lw=3)
    ax.plot(x_demo[0], y_demo[0], z_demo[0], 'k.', ms=10)
    ax.plot(x_demo[-1], y_demo[-1], z_demo[-1], 'k.', ms=10)
    ax.plot(new_traj[0, 0], new_traj[0, 1], new_traj[0, 2], 'rx', ms=10, mew=2)
    ax.plot(new_traj[-1, 0], new_traj[-1, 1], new_traj[-1, 2], 'rx', ms=10, mew=2)
    
    plt.show()

if __name__ == '__main__':
    main2D()
    #main3D()
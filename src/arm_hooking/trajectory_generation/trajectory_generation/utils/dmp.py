import numpy as np
import matplotlib.pyplot as plt

from .dmp_pastor_2009 import perform_dmp

### DMP Psuedocode

## Given: Trajectory X, Constraints C, Tuning Parameters k, D
## Output: Deformed Trajectory X' which meets constraints


#note: C is broken down into constraints and indices corresponding to those constraints in code

def DMP(X, C=None, inds=[], duration=1.0, dt=None, use_improved=True, k=None, D=None):
    return perform_dmp.perform_dmp_general(X, C, inds, duration=1.0, dt=None, use_improved=True, k=None, D=None)

def main2D():
    # demonstration
    num_points = 50
    t = np.linspace(0, 10, num_points).reshape((num_points, 1))
    x_demo = np.sin(t) + 0.01 * t**2 - 0.05 * (t-5)**2
    y_demo = np.cos(t) - 0.01 * t - 0.03 * t**2
    
    traj = np.hstack((x_demo, y_demo))
    
    new_traj = DMP(traj, [np.array([x_demo[0], y_demo[0]]), np.array([x_demo[-1], y_demo[-1]])], [0, num_points-1])
    
    new_traj2 = DMP(traj, [np.array([x_demo[0]+0.5, y_demo[0]-0.2]), np.array([x_demo[-1], y_demo[-1]])], [0, num_points-1])
    
    new_traj3 = DMP(traj, [np.array([x_demo[0], y_demo[0]]), np.array([x_demo[-1], y_demo[-1]])], [0, num_points-1], k=0.1, D=10)
    
    plt.rcParams['figure.figsize'] = (6.5, 6.5)
    fig, axs = plt.subplots(2, 2)
    axs[0][0].set_title('Demonstration')
    axs[1][0].set_title('Same Constraints')
    axs[0][1].set_title('New Initial Point')
    axs[1][1].set_title('Different Tuning (Broken?)')
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
    
    new_traj = DMP(traj, [np.array([x_demo[0] + 0.5, y_demo[0] - 0.5, z_demo[0] + 0.5]), np.array([x_demo[-1], y_demo[-1], z_demo[-1]])], [0, num_points-1])
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot(x_demo, y_demo, z_demo, 'k', lw=3)
    ax.plot(new_traj[:, 0], new_traj[:, 1], new_traj[:, 2], 'm', lw=3)
    ax.plot(x_demo[0], y_demo[0], z_demo[0], 'k.', ms=10)
    ax.plot(x_demo[-1], y_demo[-1], z_demo[-1], 'k.', ms=10)
    ax.plot(x_demo[0] + 0.5, y_demo[0] - 0.5, z_demo[0] + 0.5, 'rx', ms=10, mew=2)
    ax.plot(x_demo[-1], y_demo[-1], z_demo[-1], 'rx', ms=10, mew=2)
    
    plt.show()

if __name__ == '__main__':
    main2D()
    main3D()
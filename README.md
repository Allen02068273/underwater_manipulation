# BLUE: Behavior Learning for Underwater Environments
By Nolan Allen

## Thesis Statement
This thesis proposes a modular framework for autonomous underwater manipulation that integrates symbolic planning, learning from demonstration, optimal control, and probabilistic state estimation, enabling robust, adaptable, and data-efficient task execution in dynamic underwater environments.

## Abstract
Autonomous underwater manipulation is a challenging domain and an open problem in research, with performance often degraded by factors such as unpredictable underwater disturbances, currents, visual distortions, and sensing limitations. To this end, we propose a robust and modular underwater manipulation framework. Existing approaches often require extensive tuning or training data, use specific hardware setups, are unequipped to handle unexpected base movement and drift, or are unable to solve and generalize complex skills in novel task configurations. The inclusion of these design constraints sets our framework apart as a novel approach to address the challenges presented by practical deployment in underwater environments.

To achieve robust skill encoding and generalization, our framework integrates high-level STRIPS-based symbolic planning to generate a sequence of predefined movements, a Learning from Demonstration (LfD) model to learn basic task movements from human demonstration and generalize them to new task configurations, and an optimal control module using a Time-Varying Linear Quadratic Regulator (TVLQR) to recover smoothly from deviations in the planned trajectory while completing the movements. We further mitigate sensor noise with an Unscented Kalman Filter (UKF) to improve task pose estimation. To demonstrate our framework as learning model-agnostic, we implement three LfD models—specifically, Laplacian Trajectory Editing (LTE), Dynamic Movement Primitives (DMP), and Jerk-Accuracy (JA)—each with a unique approach. This unified and synergistic framework enables robust and efficient training and autonomous completion of a variety of underwater manipulation tasks.

We validate the framework on two real-world tasks, valve turning and loop-on-hook placement, using a Reach Alpha 5 manipulator mounted on a BlueROV underwater platform. Experimental results in laboratory and pool environments confirm the robustness, data-efficiency, adaptability, and practical potential of the framework to advance autonomous underwater manipulation. These results demonstrate our work as a useful contribution to the reliability and autonomy of underwater robotic systems.

<img width="599" height="566" alt="framework_overview" src="https://github.com/user-attachments/assets/c8c75312-ed90-4d5e-adc7-546f92b19007" />

<img width="1440" height="810" alt="robot_setup" src="https://github.com/user-attachments/assets/2bf05315-c443-4993-a65e-4975fd74cce4" />

![video_overview](https://github.com/user-attachments/assets/322e68ba-84c8-4584-a921-1fec5f12c360)

## Repository Organization
Code
- `src/` All code is contained in the this directory.
  - `arm_hooking/` The ROS2 workspace containing all the packages and nodes I created for this project. More info below.
  - `camera/` ROS2 packages used to access the camera stream from the BlueROV.
  - `planner/` Domain and task files for high-level symbolic planning.
  - `reach_robotics_sdk/` ROS2 and other code created by Reach Robotics for communicating with the Reach Alpha 5 manipulator.
  - `tools/` Various python scripts for accomplishing or automating small tasks such as cleaning the workspace, visualizing and recording data, and testing connections with the various components.
  - `unused_code/` Various python scripts that are deprecated, no longer functional, or unneeded, but are kept for future reference.
- `data` All data is contained in this directory. This includes debugging logs, training data, and graphs.

Please see `ROS2_node_overview` for a high-level description of all the ROS2 packages and nodes in the `arm_hooking/` workspace.

## Operation Instructions
Please see `operation_instructions` for the most recent information on how to run the system.

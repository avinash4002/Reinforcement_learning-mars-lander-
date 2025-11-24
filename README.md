# Reinforcement_learning-mars-rover


# Autonomous Control Systems using Reinforcement Learning (PPO)

**Project:** Mars Rover Landing & Bipedal Locomotion Simulation  
**Algorithm:** Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE)  
**Engine:** Custom PyTorch Implementation (No RL Libraries)

-----

## 1. Project Overview

This project implements the **Proximal Policy Optimization (PPO)** algorithm entirely from scratch using PyTorch. Unlike standard implementations that rely on high-level libraries like Stable-Baselines3, this codebase manually constructs the neural networks, calculates advantage estimates using Generalized Advantage Estimation (GAE), and optimizes the policy using the clipped surrogate objective function.

The system is designed to solve two contrasting physics control problems:

1.  **Discrete Control:** Mars Rover (LunarLander-v2) - Decisions are categorical (Fire Main/Left/Right).
2.  **Continuous Control:** Bipedal Walker (BipedalWalker-v3) - Decisions are continuous torque values.

-----

## 2. Methodology: Custom PPO Implementation

The core of this project is a unified **Actor-Critic** network designed to handle both discrete and continuous action spaces dynamically.



### A. The Neural Network (`ActorCritic` Class)

Instead of separate networks, we use a shared backbone with dual heads. This allows the model to learn feature representations that are useful for both acting and evaluating states.

* **Shared Backbone:**
    * `Linear(State_Dim, 64)` $\rightarrow$ `Tanh` Activation
    * `Linear(64, 64)` $\rightarrow$ `Tanh` Activation
* **Actor Head (The Policy):**
    * **Discrete Mode:** Outputs unnormalized logits. We apply a `Softmax` to get action probabilities (Categorical Distribution).
    * **Continuous Mode:** Outputs mean values ($\mu$) for action torques. We maintain a separate learnable parameter `action_var` (Log Standard Deviation) to construct a Normal Distribution ($\mathcal{N}(\mu, \sigma)$) for exploration.
* **Critic Head (The Value Function):**
    * `Linear(64, 1)`: Outputs a scalar value $V(s)$ representing the expected return of the current state.

### B. The Algorithm Logic (`PPO` Class)

The implementation handles the mathematics of Reinforcement Learning manually without external solvers:

1.  **Rollout Buffer:** A custom `Memory` class stores trajectories. Unlike DQN, PPO is on-policy, so this memory is cleared after every update.
2.  **GAE (Generalized Advantage Estimation):** We calculate advantages to reduce variance in gradient updates.
    $$A_t = \delta_t + (\gamma \lambda)A_{t+1}$$
    Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.
3.  **PPO-Clip Loss:** We implement the surrogate loss function to prevent "catastrophic forgetting" by clipping updates that deviate too far from the old policy:
    $$L = \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)$$
    * $r_t(\theta)$: Probability ratio between new and old policies.
    * $\epsilon$: Clipping parameter (set to 0.2).

-----

## 3. Deep Dive: Environments & Reward Structures

This section details the exact inputs (States), outputs (Actions), and goals (Rewards) for each simulation.

### A. Mars Rover Simulation (LunarLander-v2)

This environment simulates the "Entry, Descent, and Landing" (EDL) phase of a planetary mission.

#### **1. State Space (Observation)**
The agent receives a vector of **8 continuous values** representing the lander's physics state:

| Index | Variable | Description | Range |
| :--- | :--- | :--- | :--- |
| 0 | **Position X** | Horizontal distance from the landing pad center. | $\approx -1.5$ to $+1.5$ |
| 1 | **Position Y** | Vertical altitude (0 is the ground). | $\approx 0$ to $+1.5$ |
| 2 | **Velocity X** | Horizontal speed. | $\pm \infty$ |
| 3 | **Velocity Y** | Vertical speed (Descent rate). | $\pm \infty$ |
| 4 | **Angle** | The lander's tilt (Orientation). | Radians ($\pm \pi$) |
| 5 | **Angular Velocity** | How fast the lander is spinning. | $\pm \infty$ |
| 6 | **Left Leg Contact** | Boolean sensor (1 if touching ground, 0 if not). | 0 or 1 |
| 7 | **Right Leg Contact** | Boolean sensor (1 if touching ground, 0 if not). | 0 or 1 |

#### **2. Action Space (Discrete)**
The agent must choose exactly **one** of four actions at every timestep:

* `0`: **Do Nothing** (Freefall / Coast).
* `1`: **Fire Main Engine** (Apply vertical thrust +1.0).
* `2`: **Fire Left Engine** (Apply lateral thrust to rotate right).
* `3`: **Fire Right Engine** (Apply lateral thrust to rotate left).

#### **3. Reward Structure (The Goal)**
The total reward is the sum of the following components at every timestep:

* **Distance Penalty:** Reward increases as the lander moves closer to the landing pad $(0,0)$.
* **Velocity Penalty:** Reward is deducted if the lander is moving too fast (incentivizes slowing down).
* **Tilt Penalty:** Reward is deducted if the lander is tilted (incentivizes staying upright).
* **Leg Contact:** **+10 points** for each leg that touches the ground.
* **Engine Cost:**
    * Main Engine: **-0.03 points** per frame (incentivizes fuel efficiency).
    * Side Engines: **-0.03 points** per frame.
* **Terminal States:**
    * **Safe Landing:** **+100 points** if the lander comes to rest safely.
    * **Crash:** **-100 points** if the hull crashes into the ground.

**Solved Score:** > 200 points.

---

### B. Bipedal Locomotion (BipedalWalker-v3)

A simulation of a 4-joint robot attempting to traverse uneven terrain.

#### **1. State Space (Observation)**
The agent receives a vector of **24 continuous values**:

| Index | Variable | Description |
| :--- | :--- | :--- |
| 0 | **Hull Angle** | The robot's body tilt. |
| 1 | **Hull Angular Vel** | Rotation speed of the body. |
| 2 | **Velocity X** | Forward speed. |
| 3 | **Velocity Y** | Vertical speed (bouncing). |
| 4-7 | **Joint Angles** | Position of Hip 1, Knee 1, Hip 2, Knee 2. |
| 8-11 | **Joint Velocities** | Movement speed of the 4 joints. |
| 12 | **Leg 1 Contact** | Boolean (1 if touching ground). |
| 13 | **Leg 2 Contact** | Boolean (1 if touching ground). |
| 14-23 | **LIDAR Readings** | 10 Rangefinder sensors detecting terrain profile ahead. |

#### **2. Action Space (Continuous)**
The agent outputs a vector of **4 continuous floats** in the range $[-1.0, +1.0]$:

* `Action[0]`: Torque for **Hip 1** (Left Leg Top).
* `Action[1]`: Torque for **Knee 1** (Left Leg Bottom).
* `Action[2]`: Torque for **Hip 2** (Right Leg Top).
* `Action[3]`: Torque for **Knee 2** (Right Leg Bottom).

#### **3. Reward Structure (The Goal)**
* **Forward Progress:** Positive reward proportional to the distance traveled towards the end of the track.
* **Energy Cost:** A small penalty is applied based on the amount of motor torque used (incentivizes efficient walking, not spasming).
* **Stability Cost:** A penalty is applied if the robot's head (hull) shakes too much.
* **Failure:** **-100 points** if the robot's hull touches the ground (falls over).
* **Success:** A large bonus is awarded for reaching the far end of the track.

**Solved Score:** > 300 points.

-----

## 4. Hyperparameters

These parameters were tuned to balance stability between the two very different environments.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Optimizer** | `Adam` | Standard adaptive moment estimation optimizer. |
| **Learning Rate** | `0.0003` | Step size for weights updates. |
| **Gamma ($\gamma$)** | `0.99` | Discount factor; prioritizes long-term rewards. |
| **Lambda ($\lambda$)** | `0.95` | GAE smoothing parameter to balance bias/variance. |
| **Clip Epsilon** | `0.2` | Limits policy updates (The "Proximal" part of PPO). |
| **Update Epochs** | `10` | How many times to re-train on collected data per update. |
| **Batch Size** | `64` | Samples used per gradient descent step. |
| **Activation** | `Tanh` | Hyperbolic tangent used in hidden layers. |
| **Hidden Units** | `64` | Neurons per hidden layer. |

-----


### Quantitative Metrics

| Metric | Agent Type | Mars Rover Score (Actual) | Bipedal Walker Score (Actual) | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Reward** | **Untrained** | **-124.76** ± 41.25 | -105.0 ± 10.5 | **fail** |
| **Mean Reward** | **PPO Trained** | **+257.95** ± 17.13 | **+302.45** ± 12.5 | **Success** |
| **Sample Run** | *Video Rec.* | +226.43 | +308.12 | |
| **Improvement** | | **+382.71 points** | **+407.45 points** | |

### Qualitative Analysis

* **Mars Rover:** The agent successfully learns to stabilize angular velocity and modulate main engine thrust. It learned to "hover" slightly to kill vertical momentum before touchdown.
* **Bipedal Walker:** The agent masters the complex coordination required to balance. It developed a stable, alternating gait cycle that adapts to terrain roughness without tripping.

-----

## 6. How to Run

### Prerequisites

```bash
# Install Physics Engine and Visualization Tools
pip install gymnasium[box2d] torch numpy moviepy shimmy
# System level dependency for Box2D
sudo apt-get install -y swig

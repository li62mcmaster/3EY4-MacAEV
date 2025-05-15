# 3EY4 MacAEV – Autonomous Navigation and Mapping with ROS and LiDAR

**3EY4 MacAEV** (McMaster Autonomous Electric Vehicle) is a robotics software stack for **autonomous navigation, obstacle avoidance, and real‑time mapping** on a small Ackermann‑steered vehicle. Built on **ROS Melodic**, it fuses 2‑D LiDAR and optional depth‑camera data to drive through unknown environments while building an occupancy grid map.

---
\## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)

---

\# Features

* **LiDAR Wall Following** – keeps the vehicle centered or at a set distance from walls/hallways.
* **Gap + Virtual‑Barrier Navigation** – finds the largest free gap in the scan and solves a quadratic program to steer safely around obstacles.
* **Depth‑Camera Fusion** – augments LiDAR with RGB‑D data to spot obstacles outside the 2‑D plane (low, high, or overhanging objects).
* **Real‑Time Occupancy Mapping** – publishes a 2‑D grid map that can be visualised in RViz or used by other planners.
* **ROS‑Native & Modular** – standard messages (`sensor_msgs/LaserScan`, `ackermann_msgs/AckermannDriveStamped`, etc.) and independent nodes so you can run only what you need.

---

\## Installation

> **Tested on Ubuntu 18.04 + ROS Melodic**. For ROS Noetic, use Python 3 packages (`pip3`).

1. **Install ROS Melodic** (desktop‑full recommended). See the [ROS Wiki](http://wiki.ros.org/melodic/Installation/Ubuntu).
2. **Create a catkin workspace** (if you don’t already):

   ```bash
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```
3. **Clone the repository** into `src`:

   ```bash
   cd ~/catkin_ws/src
   git clone https://github.com/li62mcmaster/3EY4-MacAEV.git
   ```
4. **Install dependencies**:

   ```bash
   sudo apt install ros-melodic-ackermann-msgs ros-melodic-cv-bridge
   pip install numpy opencv-python quadprog
   ```
5. **Build the workspace**:

   ```bash
   cd ~/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

---

\## Usage

\### 1 ▸ Launch sensors / simulator
Ensure LiDAR publishes on `/scan` and odometry on `/odom`. For camera fusion, depth images should be on `/camera/depth/image_rect_raw`.

\### 2 ▸ Load parameters

```bash
rosparam load ~/catkin_ws/src/3EY4-MacAEV/params.yaml
```

\### 3 ▸ Run a navigation node

| Mode                      | Command                                   |
| ------------------------- | ----------------------------------------- |
| **Wall Following**        | `rosrun 3EY4-MacAEV wall_following.py`    |
| **Gap + Virtual‑Barrier** | `rosrun 3EY4-MacAEV navigation_vb.py`     |
| **LiDAR + Camera**        | `rosrun 3EY4-MacAEV navigation_vb_cam.py` |

\### 4 ▸ Visualise in RViz
Add **LaserScan**, **TF**, and **Map** displays to monitor scans, vehicle pose, and the occupancy grid.

> ⚠️ **Safety first:** Start with low speeds (`normal_speed` in `params.yaml`) and test in simulation or a safe area.

---

\## Project Structure

```
3EY4-MacAEV/
├─ wall_following.py          # LiDAR wall‑following node
├─ navigation_vb.py           # Gap + virtual‑barrier navigation
├─ navigation_vb_cam.py       # Gap‑barrier navigation with depth camera
├─ self_driving_with_camera.py# Experimental all‑in‑one script
├─ virtual_barrier.py         # Optimisation helper / alt. implementation
├─ params.yaml                # Tunable parameters & topic names
└─ README.md                  # This document
```

---

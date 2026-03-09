# Visual-Inertial Odometry (VIO)

## Overview

Visual-Inertial Odometry (VIO) is the problem of estimating the **pose of an agent** using a sequence of camera images and measurements from an **IMU (Inertial Measurement Unit)**. :contentReference[oaicite:0]{index=0}  

The system combines:

- **visual information** from a camera
- **inertial measurements** (acceleration and angular velocity)

VIO modules operate in **real time** and are widely used in robotics, autonomous navigation, and other applications. :contentReference[oaicite:1]{index=1}  

---

# Goal of the Project

The goal of this project is to implement a **basic Visual-Inertial Odometry system**.

The system will include several main components:

- smoothing / filtering of input odometry
- feature detection and descriptor computation
- image matching
- feature tracking
- 3D triangulation
- sliding-window bundle adjustment
- basic 3D reconstruction

The final system will be evaluated using public datasets based on:

- **trajectory accuracy**
- **execution speed**. :contentReference[oaicite:2]{index=2}  

---

# Pose Representation

The pose of an agent describes its full state in space:

$$
Pose = (position, orientation)
$$

Position:

$$
p = (x, y, z)
$$

Orientation can be represented using a rotation matrix:

$$
R \in SO(3)
$$

The full pose transformation can be written as:

$$
T =
\begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
$$

---

# Key Components of the System

## 1 Feature Extraction

The system extracts **keypoints** from each image frame.

Keypoints are distinctive points such as:

- corners
- intersections of edges
- textured regions

These points must be:

- repeatable
- discriminative
- spatially distributed.

---

## 2 Feature Tracking

Detected features are tracked between frames.

Tracking establishes **correspondences**:

$$
p_i^t \leftrightarrow p_i^{t+1}
$$

These correspondences allow estimation of camera motion.

---

## 3 Triangulation

Triangulation estimates the **3D coordinates of a point** from multiple observations.

If a point is observed from two camera positions:

$$
P_1, P_2
$$

its 3D position can be reconstructed using geometric constraints.

---

## 4 Bundle Adjustment

Bundle Adjustment refines both:

- camera poses
- 3D point positions

by minimizing the **reprojection error**:

$$
\min \sum ||x_{observed} - x_{projected}||^2
$$

This improves the accuracy of the reconstructed trajectory.

---

# GPU Acceleration

Some computationally expensive steps (such as feature tracking) may be accelerated using GPU technologies:

- OpenCL
- Vulkan

This allows faster processing and real-time performance.

---

# Applications

Visual-Inertial Odometry is used in:

- autonomous vehicles
- robotics
- drones
- indoor navigation systems
- space robotics.

---

# Build Instructions

The project uses **CMake**.

### Compile

```bash
./compile.sh

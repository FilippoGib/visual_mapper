# Visual Mapping Pipeline for Autonomous Racing

This repository contains the implementation of a visual mapping pipeline developed for Formula Student Driverless (FSD) autonomous racing competitions. The system performs robust traffic cone detection, real-time vehicle localization, and global mapping using stereo vision and sensor fusion techniques.

## 🚗 Overview

The pipeline consists of five core components:
1. **Cone Detection** – YOLOv11-m object detector fine-tuned on the FSOCO dataset.
2. **Visual-Inertial SLAM** – Real-time vehicle odometry from stereo + IMU.
3. **Depth Map Generation** – Stereo triangulation for per-pixel depth estimation.
4. **Fusion Module** – Projects 2D detections into 3D camera and car frames.
5. **EKF Mapping** – Builds and maintains a persistent global map of cone landmarks.

All modules are implemented as ROS 2 nodes and designed for real-time execution on embedded systems.

## 🧠 Features

- Accurate cone localization under dynamic racing conditions
- Modular ROS 2 architecture for scalability and integration
- Bird’s-eye-view visualization via image homography
- EKF-based global map refinement and outlier suppression
- Evaluation with both real-world and simulated FSD data


## 🧪 Dataset

The detector is trained on the [FSOCO dataset](https://github.com/fssic/fsoco), which includes over 11,000 annotated images of traffic cones in diverse lighting and track conditions.

## 🛠️ Setup

### Dependencies

- ROS 2 Humble (or compatible distro)
- Python 3.10+
- OpenCV
- NumPy, SciPy
- Ultralytics YOLOv8
- [ZED SDK (optional)](https://www.stereolabs.com/docs/)

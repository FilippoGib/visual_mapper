# Visual Mapping Pipeline for Autonomous Racing

This repository contains the implementation of a visual mapping pipeline developed for Formula Student Driverless (FSD) autonomous racing competitions. The system performs robust traffic cone detection, real-time vehicle localization, and global mapping using stereo vision and sensor fusion techniques.

## 🚗 Overview

The pipeline consists of five core components:
1. **Cone Detection** – YOLOv8-m object detector fine-tuned on the FSOCO dataset.
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

## 🗂️ Folder Structure

```
├── cone_detector/         # YOLOv8 ROS 2 node
├── vislam/                # Visual-inertial SLAM integration
├── depth_map/             # Stereo image processing and depth estimation
├── fusion/                # 3D point reconstruction and transformation
├── ekf_mapper/            # Global map management with EKF
├── bird_eye_view/         # Homography-based top-down projection
├── config/                # Camera intrinsics, transforms, parameters
├── launch/                # ROS 2 launch files
└── utils/                 # Calibration, visualization, helper scripts
```

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

### Install

```bash
git clone https://github.com/yourusername/fsd-mapping-pipeline.git
cd fsd-mapping-pipeline
colcon build
source install/setup.bash
```

## ▶️ Usage

```bash
ros2 launch launch/pipeline.launch.py
```

Make sure your camera driver and IMU topics are active. Use `RViz2` to visualize the output, including bounding boxes, 3D markers, and bird's-eye view overlays.

## 🧾 Citation

If you use this pipeline in academic work, please cite:

```
@misc{gibertini2025visual,
  title={A Visual Mapping Pipeline for Traffic Cone Detection and Persistent Global Mapping in Autonomous Racing},
  author={Gibertini, Filippo and Gombia, Matteo and Nels, Leonardo},
  year={2025}
}
```

## 📸 Sample Output

<div align="center">
  <img src="images/cone_detections.png" width="400"/>
  <img src="images/bird_eye_view.png" width="400"/>
</div>

## 📬 Contact

For questions or collaboration opportunities, please reach out via email:
- `300226@studenti.unimore.it`
- `302715@studenti.unimore.it`
- `270099@studenti.unimore.it`

---

*Developed at the University of Modena and Reggio Emilia for the Formula Student Driverless Competition.*

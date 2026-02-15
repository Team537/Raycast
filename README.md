## A bit of history:
Last year I attempted to build a program called OceanView, which would have helped our robot identify open scoring locations on the reef. I was still new to computer vision, and with limited access to a field it was difficult to iterate and validate results, so the project ultimately fell through.

Instead of stopping there, I started Raycast: a project meant to synthesize what I’ve learned through my time in FIRST, while still forcing me to learn new things along the way:
- **Kalman filtering**, inspired by my work attempting to reduce FTC IMU drift.
- **Pinhole camera projection**, built on ideas from OceanView and my older FTC code.
- **AI object detection**, from my very first year of FTC.
- And the countless algorithms and processing techniques I've had to learn to estimate robot centers and maintain stable tracks.

Raycast is my love letter to robotics, and I'm proud of how far it's come.

## Key Features
- **Instance-segmentation based robot detection**
  - Uses a trained YOLO26s-seg segmentation model to identify robot bumper regions per frame.
  - Produces per-instance masks + bounding boxes that feed directly into 3D position estimation and track association.
- **Robust 3D position estimation from depth + intrinsics**
  - Back-projects masked pixels into 3D camera-space using aligned depth (mm) and the RGB camera intrinsics `(fx, fy, cx, cy)`.
  - Applies robust outlier rejection using Median Absolute Deviation (MAD) gating in X/Y/Z, with a safe fallback when inliers are scarce.
  - Computes a stable center estimate via the geometric median (Weiszfeld’s algorithm with safeguards), which is resilient to depth noise and partial occlusions.
- **IMU-stabilized coordinate transforms (camera → robot → field)**
  - Uses `DepthAI` IMU rotation vectors to stabilize camera-space measurements into a consistent reference frame (anchored at IMU zero).
  - Converts stabilized vectors into robot coordinates and then into field coordinates using the robot pose streamed over the network.
- **Multi-target 3D tracking with probabilistic gating + optimal assignment**
  - Tracks robots with a constant-velocity 3D Kalman Filter state: `[x, y, z, vx, vy, vz]`. 
  - Associates detections to tracks using Mahalanobis distance (d²) with chi-square gating (DoF=3) to reject implausible matches.
  - Uses the Hungarian algorithm to compute the globally optimal assignment across all tracks/detections each frame.
  - Includes practical track management features: confirmation thresholds, missed-frame pruning, and spawn suppression to reduce duplicate tracks.
- **Periodic robot identity + alliance estimation (rate-limited for performance)** 
  - Runs `EasyOCR` team-number reads on matched tracks on a controlled cadence (with additional delay scaling based on current OCR confidence).
  - Tightens OCR regions using the segmentation mask within the detection bounding box, then applies fast deterministic preprocessing (resize, CLAHE, denoise, optional low-contrast thresholding).
  - Estimates alliance via `OpenCV` mean-color scoring on masked pixels inside the bbox (red vs blue) and keeps OCR updates stable using confidence-based overwrite rules.
- **Real-time robot I/O over the network**
  - UDP output: streams the current set of tracked robots (position + metadata) to the roboRIO.
  - UDP input: receives the robot’s field pose (translation + yaw) and frame-capture requests for debugging.
  - TCP input: receives IMU reset requests, triggering a safe IMU “zero” on the next periodic cycle.

## Required Hardware:
- OAK-D camera (we use the OAK-D Pro OV9782)
- NVIDIA Jetson Orion Nano Super (Raycast can run on other devices, but you'll likely need to adjust the code and performance settings. I'd like to make this more accessible and budget friendly over time).

**Note on AI tools**: AI-assisted tools were used as a productivity aid (e.g., summarizing documentation, reviewing approaches, and suggesting optimization ideas). All design decisions, integration, testing, and final implementation were completed and verified by me.
import os
from dotenv import load_dotenv

import time
import cv2
import numpy as np
import depthai as dai
import torch

import vision_processing.position_calculator as pose_estimator
from vision_processing.depthai_pipeline import DepthAIPipeline
from tracking.robot_tracker_3d import (RobotTracker3D, RobotDetection3D)
from file_handeling.image_saver import ImageSaver
from ultralytics.models.yolo import YOLO

from data_transmission.TCPReceiver import TCPReceiver
from data_transmission.TimeSyncServer import TimeSyncServer
from data_transmission.UDPSender import UDPRobotDetectionsSender

# Load environment variables from .env file
load_dotenv()

# ------------------------------------------------------------
# Runtime settings
# ------------------------------------------------------------
VISUALIZE_FRAMES = os.environ.get("RAYCAST_VIS", "0") == "1"
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# CUDA / Tensor Core tuning (safe defaults)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.no_grad()

# ------------------------------------------------------------
# Network settings
# ------------------------------------------------------------
RIO_IP = "192.168.55.100"
PI_IP = "192.168.55.1"

TCP_PORT = 5300
TIME_SYNC_PORT = 6000
UDP_PORT = 5800

# ------------------------------------------------------------
# Tracking configuration
# ------------------------------------------------------------
robot_tracker_3d = RobotTracker3D(
    max_missed_frames=25,     
    min_updates_to_confirm=2,
    gate_probability=.985,
    process_noise_scale=2.5,
    measurement_noise_m=0.10,
)


# ------------------------------------------------------------
# Globals (runtime state)
# ------------------------------------------------------------
depthai_pipeline: DepthAIPipeline | None = None
image_saver: ImageSaver | None = None
color_camera_intrinsics: np.ndarray | None = None

time_sync_server: TimeSyncServer | None = None
tcp_receiver: TCPReceiver | None = None
udp_sender: UDPRobotDetectionsSender | None = None

# Robot pose (field frame) from network
robot_position_field_m = np.array([0.0, 0.0, 0.0], dtype=float)
robot_yaw_field_rad = 0.0

# Yaw offset bridge: robot yaw at the instant IMU was zeroed
robot_yaw_at_imu_zero_rad = 0.0

# Flag set by network to request an IMU zero
should_zero_imu = True

# Frame capture flags (requested by network)
capture_input_frame = False
capture_output_frame = False
capture_depth_frame = False

# Time step tracking
last_frame_time_s: float | None = None

# YOLO model
model = YOLO("src/ai/best-yolo26.pt", task="segment")


# ------------------------------------------------------------
# Safe visualization helpers
# ------------------------------------------------------------
def _safe_imshow(name: str, img: np.ndarray) -> None:
    cv2.imshow(name, img)

def _safe_waitkey() -> int:
    if not VISUALIZE_FRAMES:
        return -1
    if os.environ.get("DISPLAY", "") == "":
        return -1
    return cv2.waitKey(1)


# ------------------------------------------------------------
# Debug overlay
# ------------------------------------------------------------
def overlay_instance_masks_bgr(
    frame_bgr: np.ndarray,
    masks_nhw: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay instance masks on a BGR frame.
    """
    if masks_nhw is None or len(masks_nhw) == 0:
        return frame_bgr

    out = frame_bgr.copy()
    H, W = out.shape[:2]
    N = masks_nhw.shape[0]

    hsv = np.zeros((N, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = (np.arange(N) * 180 / max(N, 1)).astype(np.uint8)
    hsv[:, 0, 1] = 200
    hsv[:, 0, 2] = 255
    bgr_colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(N, 3)

    out_f = out.astype(np.float32)
    for i in range(N):
        m = masks_nhw[i]
        if m.shape[0] != H or m.shape[1] != W:
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

        mask = m > 0.5
        if mask.sum() < 25:
            continue

        color = bgr_colors[i].astype(np.float32)
        out_f[mask] = (1.0 - alpha) * out_f[mask] + alpha * color

        # Contours + ID label (only when GUI enabled)
        if VISUALIZE_FRAMES:
            mask_u8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out_f, contours, -1, (255, 255, 255), 2)

            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(out_f, f"#{i}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return np.clip(out_f, 0, 255).astype(np.uint8)


def _display_frames(color_frame: np.ndarray, depth_mm: np.ndarray) -> None:
    cv2.imshow("Color Frame", color_frame)

    depth_colored = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_mm, alpha=0.03),
        cv2.COLORMAP_JET
    )
    cv2.imshow("Depth Frame (mm)", depth_colored)


# ------------------------------------------------------------
# Network callbacks
# ------------------------------------------------------------
def update_robot_pose(translation: tuple[float, float, float], yaw_rad: float) -> None:
    """Update robot field pose from network."""
    global robot_position_field_m, robot_yaw_field_rad
    robot_position_field_m = np.asarray(translation, dtype=float)
    robot_yaw_field_rad = float(yaw_rad)

def save_frames(save_input: bool, save_output: bool, save_depth: bool) -> None:
    """Request frame saving on the next periodic loop."""
    global capture_input_frame, capture_output_frame, capture_depth_frame
    capture_input_frame = bool(save_input) or capture_input_frame
    capture_output_frame = bool(save_output) or capture_output_frame
    capture_depth_frame = bool(save_depth) or capture_depth_frame

def zero_imu() -> None:
    """Request an IMU zero on the next periodic loop."""
    global should_zero_imu
    should_zero_imu = True

# ------------------------------------------------------------
# Robot Tracking
# ------------------------------------------------------------
def handle_no_detections(dt_s: float) -> None:
    """Handle the case when no detections are available."""
    global last_frame_time_s

    active_tracks = robot_tracker_3d.update_tracks([], dt_s, color_frame=None)

    # Debug printout
    for track in active_tracks:
        x_fwd, y_left, z_up = track.position_world_m
        print(f"[Robot {track.team_number}] X={x_fwd:.2f} Y={y_left:.2f} Z={z_up:.2f} missed={track.missed_frames}")


# ------------------------------------------------------------
# Main periodic loop
# ------------------------------------------------------------
def periodic() -> None:
    global depthai_pipeline, color_camera_intrinsics, image_saver
    global capture_input_frame, capture_output_frame, capture_depth_frame
    global should_zero_imu, robot_yaw_at_imu_zero_rad
    global last_frame_time_s, HAS_WRITTEN_TO_FILE

    if depthai_pipeline is None:
        return

    # Cache intrinsics once
    if color_camera_intrinsics is None:
        color_camera_intrinsics = depthai_pipeline.get_intrinsics(
            dai.CameraBoardSocket.CAM_A, 1280, 800
        )

    # Grab newest frames
    frames = depthai_pipeline.get_frames()
    if frames is None:
        time.sleep(0.01)
        return
    color_frame, depth_mm = frames

    # Grab newest IMU reading
    rotation_vector = depthai_pipeline.get_imu_reading()
    if rotation_vector is None:
        return

    # Handle requested IMU zero
    if should_zero_imu:
        pose_estimator.zero_imu(rotation_vector)
        robot_yaw_at_imu_zero_rad = robot_yaw_field_rad
        should_zero_imu = False

    # Save frames if requested
    if capture_input_frame and image_saver is not None:
        image_saver.save_image(color_frame, "input_frame_")
        capture_input_frame = False
 
    if capture_depth_frame and image_saver is not None:
        depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_mm, alpha=0.03), cv2.COLORMAP_JET)
        image_saver.save_image(depth_colored, "depth_frame_")
        capture_depth_frame = False

    # YOLO inference
    results = model.predict(
        source=color_frame,
        imgsz=1280,
        conf=0.10,
        iou=0.50,
        verbose=False,
        device=YOLO_DEVICE,
    )
    r0 = results[0]
    if VISUALIZE_FRAMES:
        _display_frames(color_frame, depth_mm)

    # No masks -> nothing to track
    if r0.masks is None:
        _safe_imshow("YOLO Masks Overlay", color_frame)

        # Compute dt and update tracker with no detections
        now_s = time.monotonic()
        dt_s = (1.0 / 15.0) if last_frame_time_s is None else (now_s - last_frame_time_s)
        last_frame_time_s = now_s
        handle_no_detections(dt_s)
        return

    # Pull all instance masks (N,H,W)
    masks_nhw = r0.masks.data.detach().cpu().numpy()  # type: ignore[attr-defined]
    bboxes_xyxy = r0.boxes.data.cpu().numpy() # type: ignore[attr-defined]

    # Visualization
    vis = overlay_instance_masks_bgr(color_frame, masks_nhw, alpha=0.45)
    _safe_imshow("YOLO Masks Overlay", vis)

    if capture_output_frame and image_saver is not None:
        image_saver.save_image(vis, "output_frame_")
        capture_output_frame = False

    # Build pixel lists for ALL valid masks (keep mapping to original instance id)
    objects_xy: list[np.ndarray] = []
    kept_instance_ids: list[int] = []

    for instance_id in range(masks_nhw.shape[0]):
        ys, xs = np.where(masks_nhw[instance_id] > 0.5)
        if xs.size < 50:
            continue

        pixel_points_xy = np.stack([xs, ys], axis=1).astype(np.int32)
        objects_xy.append(pixel_points_xy)
        kept_instance_ids.append(instance_id)

    if len(objects_xy) == 0:

        # Compute dt and update tracker with no detections
        now_s = time.monotonic()
        dt_s = (1.0 / 15.0) if last_frame_time_s is None else (now_s - last_frame_time_s)
        last_frame_time_s = now_s
        handle_no_detections(dt_s)
        return

    # Robust 3D positions in camera frame
    positions_camera_m = pose_estimator.robust_positions_for_all_objects_camera_m(
        objects_xy,
        depth_mm,
        color_camera_intrinsics,
        sample_step=3,
    )

    # Convert each position into field space and build detections.
    detections: list[RobotDetection3D] = []
    for (pos_cam_m, _n_used), instance_id in zip(positions_camera_m, kept_instance_ids):
        if pos_cam_m is None:
            continue

        # Convert the position to robot frame
        pos_robot_m = pose_estimator.camera_to_robot_position(
            pos_cam_m,
            rotation_vector,
            robot_yaw_at_imu_zero_rad
        )
        if pos_robot_m is None:
            continue
        
        # Convert the position to field frame
        pos_field_m = pose_estimator.robot_vector_to_field_position(
            pos_robot_m,
            robot_position_field_m,
            robot_yaw_field_rad
        )
        if pos_field_m is None:
            continue
        
        # Build detection
        pos3d = np.asarray(pos_field_m, dtype=np.float32)

        detections.append(
            RobotDetection3D(
                pos_world_m=pos3d,
                mask=masks_nhw[instance_id],
                bbox_xyxy=tuple(bboxes_xyxy[instance_id][:4]),
            )
        )

    # dt for tracker
    now_s = time.monotonic()
    dt_s = (1.0 / 15.0) if last_frame_time_s is None else (now_s - last_frame_time_s)
    last_frame_time_s = now_s

    # Update tracker
    active_tracks = robot_tracker_3d.update_tracks(detections, dt_s, color_frame=color_frame)
    
    # Debug printout
    for track in active_tracks:
        x_fwd, y_left, z_up = track.position_world_m
        print(f"[Robot {track.team_number if track.team_number != -1 else track.track_id}] X={x_fwd:.2f} Y={y_left:.2f} Z={z_up:.2f} missed={track.missed_frames}")
    
    # Send tracks over UDP
    if udp_sender is not None:
        udp_sender.send_tracks(active_tracks)

# ----------
# Setup
# ----------
if __name__ == "__main__":
    
    # Setup the vision pipeline.
    depthai_pipeline = DepthAIPipeline()
    depthai_pipeline.start_pipeline()

    # Setup the image saver.
    image_saver = ImageSaver()
    udp_sender = UDPRobotDetectionsSender(
        target_ip=RIO_IP,
        target_port=UDP_PORT,
        debug=True,
    )

    # Attempt to run the vision processing periodic loop. On program end, clean up all resources.
    try:

        while True:
           
            # Query the periodic loop.
            periodic()

            # Break the loop on 'q' key press.
            if _safe_waitkey() & 0xFF == ord('q'):
                break
    finally:

        # Cleanup resources.
        depthai_pipeline.stop_pipeline()
        
        if tcp_receiver is not None:
            tcp_receiver.stop()
        if time_sync_server is not None:
            time_sync_server.stop()
        if udp_sender is not None:
            udp_sender.close()

        if VISUALIZE_FRAMES:
            cv2.destroyAllWindows()

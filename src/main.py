import depthai as dai
import numpy as np
import time
import cv2
import random

from ultralytics.models.yolo import YOLO

from vision_processing.depthai_pipeline import DepthAIPipeline
import vision_processing.opencv_processor as cv_processor
import vision_processing.position_calculator as pose_estimator
from file_handeling.image_saver import ImageSaver

from data_transmission.TCPReceiver import TCPReceiver
from data_transmission.TimeSyncServer import TimeSyncServer

# ----------
# Settings
# ----------
# Pipeline
VISUALIZE_FRAMES = False

# OpenCV
LOWER_HSV = (4,134,118) # Example lower HSV threshold
UPPER_HSV = (48,255,255) # Example upper HSV threshold    

# ----------
# Globals
# ----------
# DepthAI
depthai_pipeline: DepthAIPipeline | None = None
color_camera_intrinsics: np.ndarray | None = None
img_saver: ImageSaver|None = None

# YOLO Object Detector
model = YOLO("src/ai/best.pt")

# Data Transmission
time_sync_server: TimeSyncServer | None = None
tcp_receiver: TCPReceiver | None = None

RIO_IP = "10.5.37.2"
PI_IP = "10.5.37.17"

UDP_PORT = 5200
TCP_PORT = 5300
TIME_SYNC_PORT = 6000

# ----------
# Data Transmission
# ----------
# Storage
camera_to_robot = { # TODO: Fill in with actual calibration data.
    "x": 0,
    "y": 0,
    "z": 0,
    "pitch": 0,
    "roll": 0,
    "yaw": 0
}

robot_pose = { # PLACEHOLDER - Will be used to convert detected object positions to world frame.
    "x": 0,
    "y": 0,
    "z": 0,
    "pitch": 0,
    "roll": 0,
    "yaw": 0
}

# Frame capture
capture_input_frame = False # Capture raw input frame
capture_output_frame = False # Capture processed output frame
capture_depth_frame = False # Capture depth frame

# ----------
# Pipeline
# ----------
def overlay_instance_masks_bgr(
    frame_bgr: np.ndarray,
    masks_nhw: np.ndarray,
    alpha: float = 0.45,
    draw_contours: bool = True,
    draw_ids: bool = True,
) -> np.ndarray:
    """
    Overlay N instance masks on a BGR frame using distinct colors.

    frame_bgr: (H, W, 3) uint8 BGR
    masks_nhw: (N, H, W) float/bool, where mask>0.5 means foreground
    """
    if masks_nhw is None or len(masks_nhw) == 0:
        return frame_bgr

    out = frame_bgr.copy()
    H, W = out.shape[:2]

    # Build a stable palette (distinct-ish colors)
    # Using HSV -> BGR for nicer separation
    N = masks_nhw.shape[0]
    hsv = np.zeros((N, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = (np.arange(N) * 180 / max(N, 1)).astype(np.uint8)  # hue
    hsv[:, 0, 1] = 200  # saturation
    hsv[:, 0, 2] = 255  # value
    bgr_colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(N, 3)

    for i in range(N):
        m = masks_nhw[i]
        if m.shape[0] != H or m.shape[1] != W:
            # If mask size differs, resize to frame
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

        mask = (m > 0.5)

        if mask.sum() < 25:
            continue

        color = bgr_colors[i].astype(np.float32)

        # Alpha-blend only where mask is true:
        # out[mask] = (1-a)*out + a*color
        out_f = out.astype(np.float32)
        out_f[mask] = (1.0 - alpha) * out_f[mask] + alpha * color
        out = np.clip(out_f, 0, 255).astype(np.uint8)

        if draw_contours:
            mask_u8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (255, 255, 255), 2)

        if draw_ids:
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(out, f"#{i}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return out

def display_frames(color_frame, depth_frame_mm):
    """
    Display the color and depth frames using OpenCV.

    :param color_frame (np.ndarray): The color (RGB) frame.
    :param depth_frame_mm (np.ndarray): The depth frame in millimeters.
    """
    cv2.imshow("Color Frame", color_frame)

    # Apply a colormap to the depth frame for better visualization.
    depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame_mm, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imshow("Depth Frame (mm)", depth_colored)

def periodic():
    """
    Periodic function to be called in the main loop.    
    """
    # Bound the variables.
    global depthai_pipeline
    global color_camera_intrinsics
    global img_saver
    global capture_input_frame, capture_output_frame, capture_depth_frame
    global model

    # Verify that everything has been setup.
    if depthai_pipeline is None:
        return
    
    # Grab the color camera intrinsics if not already done.
    if color_camera_intrinsics is None:
        color_camera_intrinsics = depthai_pipeline.get_intrinsics(
            dai.CameraBoardSocket.CAM_A,
            1280,
            800
        )

    # Get the most recent color and depth frames.
    frames = depthai_pipeline.get_frames()
    if frames is None:
        time.sleep(0.05) # Prevent busy-waiting.
        return
    color_frame, depth_mm = frames
     
    # Save the input frame if requested.
    if capture_input_frame and img_saver is not None:
        img_saver.save_image(color_frame, "input_frame_")
        capture_input_frame = False

    # Save the depth frame if requested.
    if capture_depth_frame and img_saver is not None:
        depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_mm, alpha=0.03), cv2.COLORMAP_JET)
        img_saver.save_image(depth_colored, "depth_frame_")
        capture_depth_frame = False

    # Run YOLO inference on the color frame.
    results = model.predict(
        source=color_frame,
        imgsz=640,
        conf=0.15,
        iou=0.5,
        verbose=False
    )

    # Process the results.
    r0 = results[0]
    if r0.masks is None:
        cv2.imshow("YOLO Masks Overlay", color_frame)
        return
    
    vis = color_frame
    if r0.masks is not None:
        masks = r0.masks.data.cpu().numpy()  # pyright: ignore[reportAttributeAccessIssue] # (N, H, W)
        vis = overlay_instance_masks_bgr(color_frame, masks, alpha=0.45, draw_contours=True, draw_ids=True)
        cv2.imshow("YOLO Masks Overlay", vis)
    else:
        cv2.imshow("YOLO Masks Overlay", color_frame)

    # Save the output frame if requested.
    if capture_output_frame and img_saver is not None:
        img_saver.save_image(vis, "output_frame_")
        capture_output_frame = False

    # r0.masks.data: (N, H, W) torch tensor (bool/float)
    masks = r0.masks.data.cpu().numpy() # pyright: ignore[reportAttributeAccessIssue]
    objects_xy = []

    # Extract the pixel coordinates of each detected object's mask.
    for k in range(masks.shape[0]):
        ys, xs = np.where(masks[k] > 0.5)
        if len(xs) < 50:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.int32)  # (Npts, 2) in pixel coords
        objects_xy.append(pts)

    # Compute / Estimate the position of the object.
    results = pose_estimator.robust_positions_for_all_objects_camera_m(objects_xy, depth_mm, color_camera_intrinsics)

    for i, (pos, n) in enumerate(results):
        if pos is None:
            continue
        X, Y, Z = pos  # meters, camera frame
        print(f"Obj {i}: X={X:.3f}m Y={Y:.3f}m Z={Z:.3f}m (pixels used={n})")

    # Display the frames to the user to aid debugging. (If enabled)
    if VISUALIZE_FRAMES:
        display_frames(color_frame, depth_mm)

# ----------
# Data Transmission
# ----------
def update_robot_pose(pose):
    robot_pose = pose

def save_frames(save_input_frame, save_output_frame, save_depth_frame):
    """
    Saves the specified frames to the file.
    """

    # Use "or" to retain previous settings if None.
    global capture_input_frame, capture_depth_frame, capture_output_frame
    capture_input_frame = save_input_frame or capture_input_frame
    capture_output_frame = save_output_frame or capture_output_frame
    capture_depth_frame = save_depth_frame or capture_depth_frame

# ----------
# Setup
# ----------
if __name__ == "__main__":

    # Setup the vision pipeline.
    depthai_pipeline = DepthAIPipeline()
    depthai_pipeline.start_pipeline()

    # Setup the image saver.
    img_saver = ImageSaver()

    # Setup data transmission 
    time_sync_server = TimeSyncServer(
        ip=PI_IP,
        port=TIME_SYNC_PORT,
        timeout_s=1.0,
        debug=True
    )

    tcp_receiver = TCPReceiver(
        update_robot_pose=update_robot_pose,
        save_frames=save_frames,
        ip=PI_IP,
        port=TCP_PORT,
        debug=True
    )

    # Start data transmission servers.
    time_sync_server.start()
    tcp_receiver.start()

    try:
        while True:
           
            # Query the periodic loop.
            periodic()

            # Break the loop on 'q' key press.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:

        # Cleanup resources.
        depthai_pipeline.stop_pipeline()
        cv2.destroyAllWindows()
        
        if tcp_receiver is not None:
            tcp_receiver.stop()
        if time_sync_server is not None:
            time_sync_server.stop()
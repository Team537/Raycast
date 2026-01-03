import depthai as dai
import numpy as np
import time
import cv2

from vision_processing.depthai_pipeline import DepthAIPipeline
import vision_processing.opencv_processor as cv_processor
import vision_processing.position_calculator as pose_estimator

# ----------
# Settings
# ----------
# Pipeline
VISUALIZE_FRAMES = True

# OpenCV
LOWER_HSV = (89,176,41)  # Example lower HSV threshold
UPPER_HSV = (107,255,194) # Example upper HSV threshold    

# ----------
# Globals
# ----------
# DepthAI
depthai_pipeline: DepthAIPipeline | None = None
color_camera_intrinsics: np.ndarray | None = None

# ----------
# Pipeline
# ----------
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

    # Verify that everything has been setup.
    if depthai_pipeline is None:
        return
    
    # Grab the color camera intrinsics if not already done.
    if color_camera_intrinsics is None:
        color_camera_intrinsics = depthai_pipeline.get_intrinsics(
            dai.CameraBoardSocket.RGB,
            1280,
            800
        )

    # Get the most recent color and depth frames.
    frames = depthai_pipeline.get_frames()
    if frames is None:
        time.sleep(0.05) # Prevent busy-waiting.
        return
    color_frame, depth_mm = frames
     
    # Threshold the image.
    threshold_frame, _ = cv_processor.mask_image(color_frame, LOWER_HSV, UPPER_HSV)
    cv2.imshow("Threshold Frame", threshold_frame)

    # Extract objects from the thresholded frame.
    objects_xy, object_mask = cv_processor.extract_objects(threshold_frame)

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
# Setup
# ----------
if __name__ == "__main__":

    # Setup the vision pipeline.
    depthai_pipeline = DepthAIPipeline()
    depthai_pipeline.start_pipeline()

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
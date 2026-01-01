import depthai as dai
import numpy as np
import time
import cv2

from vision_processing.depthai_pipeline import DepthAIPipeline

# ----------
# Variables
# ----------
# DepthAI

# ----------
# Utility
# ----------
def mm_to_meters(depth_frame_mm):
    """
    Convert a depth frame from millimeters to meters.

    Args:
        depth_frame_mm (np.ndarray): Depth frame in millimeters.

    Returns:
        np.ndarray: Depth frame in meters.
    """
    return depth_frame_mm.astype(np.float32) / 1000.0  

# ----------
# Pipeline
# ----------
def display_frames(color_frame, depth_frame_mm):
    """
    Display the color and depth frames using OpenCV.

    Args:
        color_frame (np.ndarray): The color (RGB) frame.
        depth_frame_mm (np.ndarray): The depth frame in millimeters.
    """
    cv2.imshow("Color Frame", color_frame)

    # Apply a colormap to the depth frame for better visualization.
    depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame_mm, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imshow("Depth Frame (mm)", depth_colored)

def periodic():
    """
    Periodic function to be called in the main loop.
    """
    # Verify that everything has been setup.
    if depthai_pipeline is None:
        return
    
    # Get the most recent color and depth frames.
    color_frame = depthai_pipeline.get_color_frame()
    depth_frame = depthai_pipeline.get_depth_frame_mm()

    # Verify that frames were retrieved.
    if color_frame is None or depth_frame is None:
        time.sleep(0.05) # Prevent busy-waiting.
        return
    
    # Display the frames to the user to aid debugging.
    display_frames(color_frame, depth_frame)

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
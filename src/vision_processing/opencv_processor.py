import cv2 
import numpy as np
from typing import List, Dict, Any, Tuple

# Constants
black_hsv = np.array([[[1,1,1]]], dtype=np.uint8)
white_hsv = np.array([[[255,255,255]]], dtype=np.uint8)

# ---------- 
# Helpers 
# ---------- 
def mask_image(
            frameBGR: cv2.typing.MatLike,
            lower_hsv,
            upper_hsv
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a mask from the image, isolating pixels within a certain color range.
    
    :param frameBGR: The input BGR frame to be masked.
    :param lower_hsv: The lower HSV threshold.
    :param upper_hsv: The upper HSV threshold.
    """
    frameHSV = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frameHSV, lower_hsv, upper_hsv)
    masked_frame = cv2.bitwise_and(frameBGR, frameBGR, mask=mask)
    return masked_frame, mask

def visualize_object_points(
            objects_xy, 
            shape_hw, 
            overlay_on=None, 
            alpha=0.6
        ):
    """
    Visualize object points by coloring each object's pixels distinctly.
    \n (Written by ChatGPT with minor modifications)

    :param objects_xy: list of (N,2) arrays of (x,y)
    :param shape_hw: (h,w)
    :param overlay_on: optional BGR image to overlay on (same size)
    :param alpha: overlay strength if overlay_on is provided
    """
    h, w = shape_hw
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # Simple palette (repeats if > len(palette) objects)
    palette = [
        (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (180, 180, 255), (180, 255, 180), (255, 180, 180)
    ]

    for i, xy in enumerate(objects_xy):
        if xy.size == 0:
            continue
        color = palette[i % len(palette)]
        xs = xy[:, 0]
        ys = xy[:, 1]
        # Paint each pixel belonging to this object
        vis[ys, xs] = color

    if overlay_on is not None:
        base = overlay_on.copy()
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(base, 1.0, vis, alpha, 0)
        return out, vis

    return vis, vis

def extract_objects(
            mask: np.ndarray, 
            peel_px: int = 1, 
            min_area: int = 100,
            visualize_mask: bool = True,
            visualize_point_clouds: bool = True,
        ):
    """
    Extract objects from a binary mask using contours.

    :param mask: The mask used to extract objects.
    :param peel_px: The number of pixels to peel from the edges.
    :param min_area: Minimum area to consider a contour as an object.

    :return: A list of objects, each represented as an array of (x,y) pixel coordinates.
    :rtype: List[np.ndarray]
    """
    # 2) Light morphology to remove specks / close tiny gaps (tune as needed).
    kernel = np.ones((3, 3), np.uint8)
    object_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # Remove tiny specks
    object_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Close tiny gaps

    # Peel edges inward by peel_px using erosion (shrinks object boundary)
    if peel_px > 0:
        peel_kernel = np.ones((2 * peel_px + 1, 2 * peel_px + 1), np.uint8)
        object_mask = cv2.erode(object_mask, peel_kernel, iterations=1)

    # 3) Extract object points from the image.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((object_mask > 0).astype(np.uint8), connectivity=8)

    objects_xy = []
    for label in range(1, num_labels):  # label 0 is background
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        ys, xs = np.where(labels == label)
        xy = np.column_stack((xs, ys)).astype(np.int32)  # (x, y)
        objects_xy.append(xy)

    # 4) Visualization (optional)
    if visualize_mask:
        cv2.imshow("Object Mask", object_mask)
    if visualize_point_clouds:
        h, w = object_mask.shape[:2]
        overlay, vis = visualize_object_points(objects_xy, (h, w), overlay_on=mask, alpha=0.7)

        cv2.imshow("Points (overlay)", overlay)

    return objects_xy, object_mask
from enum import Enum, auto
import cv2
from matplotlib import image
from matplotlib.pyplot import box
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

import easyocr
from ultralytics.models.yolo import YOLO
import time

# ----------------------------------------------------------
# OpenOCR Model
# ----------------------------------------------------------
reader = easyocr.Reader(["en"], gpu=True)

ALLOWED_CHARACTERS = "0123456789" # Only numbers.
MIN_CONFIDENCE = 0.15

# ----------------------------------------------------------
# Color Constants
# ----------------------------------------------------------
RED_BGR = (0, 0, 255)
BLUE_BGR = (255, 0, 0)
class RobotColor(Enum):
    RED = "red"
    BLUE = "blue"
    UNKNOWN = "unknown"

def compute_color_score_bgr(mean_color: tuple[float, float, float], target_color: tuple[int, int, int]) -> float:
    """
    Computes the euclidean color distance score, detailing how far away mean_color is from target_color. This function expects
    the values to be provided in the BGR color format.

    :param mean_color: The average color of the image.
    :param target_color: The color you are trying to evaluate closeness to.
    :return: The euclidean color distance score.
    :rtype: float
    """
    b_score = abs(target_color[0] - mean_color[0])
    g_score = abs(target_color[1] - mean_color[1])
    r_score = abs(target_color[2] - mean_color[2])

    return b_score + g_score + r_score

# ----------------------------------------------------------
# Robot
# ----------------------------------------------------------
def _tighten_roi_with_mask(
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    pad: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (roi_bgr, roi_mask_u8) where ROI is tightened to mask pixels within bbox.
    mask_u8 must be 0/255 and same HxW as img_bgr.
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy

    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W,     x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H,     y2))
    if x2 <= x1 or y2 <= y1:
        return img_bgr[0:1, 0:1], mask_u8[0:1, 0:1]

    roi_img = img_bgr[y1:y2, x1:x2]
    roi_mask = mask_u8[y1:y2, x1:x2]

    ys, xs = np.where(roi_mask > 0)
    if len(xs) < 50:  # not enough mask pixels; fall back to bbox ROI
        return roi_img, roi_mask

    mx1, mx2 = xs.min(), xs.max()
    my1, my2 = ys.min(), ys.max()

    # pad around the mask region
    mx1 = max(0, mx1 - pad)
    my1 = max(0, my1 - pad)
    mx2 = min(roi_img.shape[1] - 1, mx2 + pad)
    my2 = min(roi_img.shape[0] - 1, my2 + pad)

    roi_img = roi_img[my1:my2 + 1, mx1:mx2 + 1]
    roi_mask = roi_mask[my1:my2 + 1, mx1:mx2 + 1]
    return roi_img, roi_mask

def preprocess_for_easyocr_fast_safe(
    roi_bgr: np.ndarray,
    roi_mask_u8: Optional[np.ndarray] = None,
    target_height: int = 160
) -> np.ndarray:
    """
    Single-pass preprocessing designed for EasyOCR:
    - Prefer grayscale (preserves stroke detail)
    - Mild CLAHE + gentle denoise
    - Only threshold if contrast is extremely low (deterministic, no retries)
    Returns an 8-bit single-channel image (grayscale OR binary).
    """
    if roi_mask_u8 is not None:
        roi_mask_u8 = (roi_mask_u8 > 0).astype(np.uint8) * 255
        roi_bgr = cv2.bitwise_and(roi_bgr, roi_bgr, mask=roi_mask_u8)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    # Resize to consistent height (helps OCR while staying fast)
    h, w = gray.shape[:2]
    if h > 0 and h != target_height:
        scale = target_height / float(h)
        new_w = max(1, int(round(w * scale)))
        gray = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
        if roi_mask_u8 is not None:
            roi_mask_u8 = cv2.resize(roi_mask_u8, (new_w, target_height), interpolation=cv2.INTER_NEAREST)

    # Mild local contrast normalization (keep conservative)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Gentle denoise that keeps edges (fast and safe)
    # Median 3x3 tends to remove speckle without smearing digits much.
    gray = cv2.medianBlur(gray, 3)

    # If we have a mask, keep only masked region (helps detector)
    if roi_mask_u8 is not None:
        # Slight dilate to include digit edges near mask boundary
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        m = cv2.dilate(roi_mask_u8, k, iterations=1)
        gray = cv2.bitwise_and(gray, gray, mask=m)

    # ---- Deterministic "only if needed" binarization ----
    # Compute contrast on non-zero pixels (masked region) if possible.
    if roi_mask_u8 is not None:
        vals = gray[roi_mask_u8 > 0]
        if vals.size > 100:
            vals_f = np.asarray(vals, dtype=np.float64)
            contrast = float(np.percentile(vals_f, 90) - np.percentile(vals_f, 10))
        else:
            gray_f = np.asarray(gray, dtype=np.float64)
            contrast = float(np.percentile(gray_f, 90) - np.percentile(gray_f, 10))
    else:
        gray_f = np.asarray(gray, dtype=np.float64)
        contrast = float(np.percentile(gray_f, 90) - np.percentile(gray_f, 10))

    # If extremely low contrast, thresholding can help; otherwise keep grayscale.
    if contrast < 35.0:
        # Adaptive threshold is less destructive than Otsu in uneven lighting
        block = 31 if min(gray.shape[:2]) >= 80 else 15
        bin_img = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block, 5
        )
        return bin_img
    
    return gray

def ocr_extract_team_number(
    reader,
    img_bgr: np.ndarray,
    mask_u8: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int]
) -> tuple[int, float]:

    # Ensure mask is HxW uint8 {0,255} and matches img size. 
    # This is crucial for the tightening step and OCR focus. I
    # f the mask is not properly aligned or scaled to the image, the subsequent steps will fail or produce poor results. 
    # The function is robust to common issues like float masks in [0,1] or [0,255], and mismatched sizes, which can happen due to model output variations or preprocessing steps.
    H, W = img_bgr.shape[:2]
    if mask_u8.shape[:2] != (H, W):
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    if mask_u8.dtype != np.uint8:
        mask_u8 = (mask_u8 > (0.5 if mask_u8.max() <= 1.0 else 127)).astype(np.uint8) * 255
    else:
        mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255

    # Tighten the ROI to the masked region within the bounding box. 
    # This helps focus OCR on the relevant area and can improve accuracy, especially if the initial bbox is loose.
    #  The padding ensures we don't cut off digit edges, which is important for recognition. 
    # The output is a smaller image that contains the team number with less background noise.
    roi_bgr, roi_mask = _tighten_roi_with_mask(img_bgr, mask_u8, bbox_xyxy, pad=8)

    # Preprocess the ROI for OCR. This includes resizing, contrast enhancement, and optional masking to focus on the digit region.
    #  The preprocessing is designed to be fast and deterministic, avoiding any iterative retries. 
    # The output is a single-channel image that is suitable for EasyOCR's text detection and recognition stages.
    pre = preprocess_for_easyocr_fast_safe(roi_bgr, roi_mask_u8=roi_mask, target_height=160)

    # EasyOCR can be a bit slow, so we use the tuned parameters to balance speed and accuracy. The allowlist ensures we only get digits, which is our expected team number format. We also set a low text threshold to catch faint digits, but rely on the confidence score to filter out false positives.
    results = reader.readtext(
        pre,
        allowlist=ALLOWED_CHARACTERS,   
        detail=1, # Needed to get confidence estimate  
        paragraph=False,
        decoder="beamsearch",         
        beamWidth=2,
        text_threshold=0.5,        
        low_text=0.15,
        link_threshold=0.35,
        # let EasyOCR handle contrast internally if it wants
        contrast_ths=0.1,
        adjust_contrast=0.5,
        batch_size=64
    ) 

    # Parse results to find the best team number candidate based on confidence. 
    # We filter out any non-digit results and keep track of the highest confidence valid prediction. 
    # The final output is the estimated team number and its associated confidence score, which can be used downstream for decision-making or display.
    best_num = -1
    best_confidence = 0
    for (_bb, text, confidence) in results:
        if not isinstance(confidence, float):
            continue
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            continue
        if confidence > best_confidence and confidence >= MIN_CONFIDENCE:
            best_confidence = confidence
            best_num = int(digits)

    return best_num, best_confidence

def extract_robot_info(color_img: np.ndarray, robot_mask: np.ndarray, bounding_box: tuple[float, float, float, float]):
    """
    :param: color_img: BGR image (H,W,3)
    :param: robot_mask: YOLO mask, typically float32 (H,W) in [0,1] or [0,255]
    :param: bounding_box: expected xyxy (x1,y1,x2,y2)
    """
    global reader

    H, W = color_img.shape[:2]

    # ---------------------------
    # 0) Ensure bbox is valid xyxy and clamp
    # ---------------------------
    x1, y1, x2, y2 = map(int, map(round, bounding_box))
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W,     x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H,     y2))

    if x2 <= x1 or y2 <= y1:
        return (RobotColor.RED, -1, 0.0)

    # ---------------------------
    # 1) Ensure mask is HxW uint8 {0,255}
    # ---------------------------
    mask_hw = robot_mask
    if mask_hw.ndim == 3:
        # just in case a singleton channel exists
        mask_hw = mask_hw.squeeze()

    # Ultralytics usually matches the input image size, but guard anyway.
    if mask_hw.shape[:2] != (H, W):
        mask_hw = cv2.resize(mask_hw, (W, H), interpolation=cv2.INTER_NEAREST)

    # Convert float/probability mask to uint8 binary mask
    if mask_hw.dtype != np.uint8:
        # If values are 0..1, threshold at 0.5; if 0..255, threshold at 127.
        thresh = 0.5 if mask_hw.max() <= 1.0 else 127
        mask_u8 = (mask_hw > thresh).astype(np.uint8) * 255
    else:
        # If already u8, make sure it's binary-ish
        mask_u8 = (mask_hw > 0).astype(np.uint8) * 255

    # ---------------------------
    # 2) Compute mean color *inside bbox* to avoid background
    # ---------------------------
    cropped_img = color_img[y1:y2, x1:x2]
    cropped_mask = mask_u8[y1:y2, x1:x2]

    # OpenCV expects mask to be 8-bit, same HxW as image ROI
    mean_b, mean_g, mean_r, _ = cv2.mean(cropped_img, mask=cropped_mask)
    mean_color_bgr = (mean_b, mean_g, mean_r)

    red_score = compute_color_score_bgr(mean_color_bgr, RED_BGR)
    blue_score = compute_color_score_bgr(mean_color_bgr, BLUE_BGR)

    # Pick the closer one (smaller score)
    robot_color = RobotColor.RED if red_score < blue_score else RobotColor.BLUE

    # ---------------------------
    # 3) OCR for team number (use ROI; optionally mask it)
    # ---------------------------
    t1 = time.time()

    # Call OCR on full-frame + full-frame mask + full-frame bbox.
    xyxy = (x1, y1, x2, y2)
    team, confidence = ocr_extract_team_number(reader, color_img, mask_u8, xyxy)
    t2 = time.time()
    print(f"OCR took {(t2 - t1)*1000:.2f} ms")
    return (robot_color, team, confidence)

# ----------------------------------------------------------
# OCR Warmup
# ----------------------------------------------------------
warmup_directory = Path("src/storage/ocr_warmpu")
if warmup_directory.is_dir():

    # Preprocess all provided image frames to speed up the first few real detection.
    # This is likely to be done before the match begins, so it basically loads the model into memory.
    for warmup_photo in warmup_directory.iterdir():

        # Extract info from each image. This will be output for debugging sake.        
        t1 = time.time()

        results = reader.readtext(
            cv2.imread(str(warmup_photo.absolute())),
            allowlist=ALLOWED_CHARACTERS,   
            detail=1, # Needed to get confidence estimate  
            paragraph=False,
            decoder="beamsearch",         
            beamWidth=2,
            text_threshold=0.5,        
            low_text=0.15,
            link_threshold=0.35,

            # let EasyOCR handle contrast internally if it wants
            contrast_ths=0.1,
            adjust_contrast=0.5,
            batch_size=16
        )

        # Print the result of the warmup frame processing. This is done to show progress in the terminal.
        t2 = time.time()
        print(f"Warmup frame processed! OCR took {(t2 - t1)*1000:.2f} ms \n" + str(results)) 
        
# ----------------------------------------------------------
# Testing
# ----------------------------------------------------------
if __name__ == "__main__":

    # Load the YOLO model
    model = YOLO("src/ai/best-yolo26.pt", task="segment")

    file_path = "src/Img5.png"

    # Read the image.
    test_image = cv2.imread(file_path)
    
    # Predict the robots in the image and then run the test classifier.
    results = model.predict(
        source=file_path, 
        imgsz=1280,
        conf=0.15,
        iou=0.50,
        verbose=False,
        device="cpu"
    )

    r0 = results[0]

    # No masks -> nothing to track
    if r0.masks is None:
        pass

    # Pull all instance masks (N,H,W) * boxes (x,y,w,h)
    masks_nhw = r0.masks.data.detach().cpu().numpy()  # type: ignore[attr-defined]
    bboxes_xyxy_conf_cls = r0.boxes.data.cpu().numpy() # type: ignore[attr-defined]

    # Extract all valid information.
    for mask, bbox in zip(masks_nhw, bboxes_xyxy_conf_cls):

        # Display debug info for the robot results.
        robot_color, est_team_num, highest_confidence = extract_robot_info(test_image, mask, bbox[:4]) # pyright: ignore[reportArgumentType]
        print(f"Robot Details: [Color: {robot_color}] [Team #: {est_team_num} - Confidence: {highest_confidence}")
from typing import Optional
import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation as R

# ----------------------------
# Frame conventions used here
# ----------------------------
# Camera:
#   +X = right, +Y = down, +Z = forward
#
# IMU axes:
#   +X = down, +Y = toward right camera (device right), +Z = backward
#
# World:
#   +X = forward, +Y = left, +Z = up

# Camera -> IMU/body axis mapping matrix (R_B_from_C)
# v_B = [down, right, back] = [v_cam_y, v_cam_x, -v_cam_z]
R_B_FROM_C = np.array([
    [0.0, 1.0,  0.0],   # Bx (down)  = Cam Y (down)
    [1.0, 0.0,  0.0],   # By (right) = Cam X (right)
    [0.0, 0.0, -1.0],   # Bz (back)  = -Cam Z (forward)
], dtype=float)

# Body(ref) -> World mapping (R_W_from_B0)
# world forward = -body back, world left = -body right, world up = -body down
R_W_FROM_B0 = np.array([
    [0.0, 0.0, -1.0],   # Wx (forward) = -Bz (back)
    [0.0, -1.0, 0.0],   # Wy (left)    = -By (right)
    [-1.0, 0.0, 0.0],   # Wz (up)      = -Bx (down)
], dtype=float)

# Camera position in ROBOT frame (meters).
# Robot frame: +X forward, +Y left, +Z up
CAMERA_POS_IN_ROBOT_M = np.array([
    0.00,   # +X forward  (e.g., 20 cm in front of robot origin)
    0.00,   # +Y left
    0.00,   # +Z up       (e.g., 45 cm above robot origin)
], dtype=float)

CAMERA_YAW_IN_ROBOT_RAD = np.deg2rad(0)  # TODO: measure this

# Storage for the IMU offset.
q_ref = None 


# ----------------------------
# World -> Robot Helpers
# ----------------------------
def rotz(rad: float) -> R:
    """Yaw rotation about +Z (up)."""
    return R.from_euler("z", rad, degrees=False)  # SciPy supports from_euler/apply/inv :contentReference[oaicite:2]{index=2}

def stabilized_world_vector_to_robot_vector(
    object_vector_stabilized_world_m: np.ndarray,
    yaw_robot_in_stabilized_world_rad: float,
) -> np.ndarray:
    """
    Convert a stabilized-world vector (your X fwd, Y left, Z up) into ROBOT coordinates.

    Convention:
      yaw_robot_in_stabilized_world_rad = angle from stabilized_world +X to robot +X (CCW about +Z)

    If v_w = R_world_from_robot * v_robot, then v_robot = (R_world_from_robot)^-1 * v_w.
    """
    R_world_from_robot = rotz(yaw_robot_in_stabilized_world_rad)
    v_robot = R_world_from_robot.inv().apply(object_vector_stabilized_world_m)
    return v_robot

def robot_vector_to_field_position(
    object_vector_robot_m: np.ndarray,
    robot_position_field_m: np.ndarray,
    yaw_robot_in_field_rad: float,
) -> np.ndarray:
    """
    Convert a robot-frame relative vector into an absolute field position.

    :param object_vector_robot_m: (3,) float32 vector in robot frame
    :param robot_position_field_m: (3,) float32 robot position in field frame
    :param yaw_robot_in_field_rad: robot yaw in field frame (CCW about +Z)
    :return: (3,) float32 object position in field frame
    """
    R_field_from_robot = rotz(yaw_robot_in_field_rad)
    p_field_obj = robot_position_field_m + R_field_from_robot.apply(object_vector_robot_m)
    return p_field_obj

def camera_to_robot_position(
    pos_cam_m: np.ndarray,
    rotation_vector,
    yaw_robot_in_stabilized_world_rad: float,
) -> np.ndarray | None:
    """
    Returns object position in ROBOT frame (meters), relative to robot origin.

    Robot frame: +X forward, +Y left, +Z up
    """
    # 1) Object vector in "stabilized world" axes (anchored at IMU zero)
    v_stabilized = camera_to_world(pos_cam_m, rotation_vector)
    if v_stabilized is None:
        return None

    # 2) Convert stabilized-world vector -> robot vector using yaw at IMU zero
    # (this is the missing bridge in your current code)
    v_robot = stabilized_world_vector_to_robot_vector(
        v_stabilized,
        yaw_robot_in_stabilized_world_rad
    )

    # 3) Apply fixed camera mounting yaw (if your camera optical axes are yawed vs robot)
    # Note: only keep this if CAMERA_YAW_IN_ROBOT_RAD is truly needed.
    v_robot = rotz(CAMERA_YAW_IN_ROBOT_RAD).apply(v_robot)

    # 4) Translate from camera origin to robot origin
    return CAMERA_POS_IN_ROBOT_M + v_robot


# ----------------------------
# DepthAI IMU helpers
# ----------------------------
def zero_imu(rotation_vector) -> None:
    """Set current IMU orientation as reference."""
    global q_ref
    q_ref = R.from_quat([rotation_vector.i, rotation_vector.j, rotation_vector.k, rotation_vector.real])

def is_imu_zeroed() -> bool:
    return q_ref is not None

def get_relative_rotation(rotation_vector) -> R | None:
    """
    Returns R_rel = R_ref^{-1} * R_cur.
    This maps vectors from the CURRENT body frame into the REFERENCE body frame.
    """
    if q_ref is None:
        return None
    q_cur = R.from_quat([rotation_vector.i, rotation_vector.j, rotation_vector.k, rotation_vector.real])
    return q_ref.inv() * q_cur

def camera_to_world(pos_cam: np.ndarray, rotation_vector) -> np.ndarray | None:
    """
    Convert a 3D point from camera optical frame (X right, Y down, Z forward)
    into a stabilized world frame (X forward, Y left, Z up), using IMU relative rotation.
    """
    q_rel = get_relative_rotation(rotation_vector)
    if q_rel is None:
        return None

    # Camera -> current Body/IMU
    v_b = R_B_FROM_C @ pos_cam

    # Stabilize: current body -> reference body (IMU-zero)
    v_b0 = q_rel.apply(v_b)

    # Reference body -> stabilized world (forward/left/up)
    return R_W_FROM_B0 @ v_b0
    
def depthai_quat_to_rot(rotation_vector) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert DepthAI rotation vector to rotation matrix and Euler angles (degrees).
    """
    q_xyzw = np.array([rotation_vector.i, rotation_vector.j, rotation_vector.k, rotation_vector.real], dtype=float)
    rot = R.from_quat(q_xyzw)  # scalar-last by default
    R_cam = rot.as_matrix()    # 3x3 rotation matrix
    euler_rpy = rot.as_euler("xyz", degrees=True)  # roll, pitch, yaw (convention choice!)
    return R_cam, euler_rpy


# ----------------------------
# Robust stats helpers
# ----------------------------
def _mad(x: np.ndarray) -> float:
    """
    Median Absolute Deviation (robust scale).
    """
    med = np.median(x)
    return float(np.median(np.abs(x - med))) + 1e-12

def geometric_median(points: np.ndarray, eps: float = 1e-6, max_iter: int = 128) -> np.ndarray:
    """
    Geometric median of N 3D points (minimizes sum of Euclidean distances).
    Very robust to outliers.
    
    Weiszfeld's algorithm with safeguards.
    """
    if points.shape[0] == 0:
        raise ValueError("No points for geometric_median")

    # Start at coordinate-wise median (robust initializer)
    y = np.median(points, axis=0)

    for _ in range(max_iter):
        d = np.linalg.norm(points - y, axis=1)
        # If we hit an existing point (distance ~0), that's already a minimizer
        if np.any(d < eps):
            return points[np.argmin(d)]

        w = 1.0 / np.maximum(d, eps)
        y_next = (points * w[:, None]).sum(axis=0) / w.sum()

        if np.linalg.norm(y_next - y) < eps:
            return y_next
        y = y_next

    return y


# ----------------------------
# Main function: per-object 3D aggregation
# ----------------------------
def robust_object_position_camera_m(
    xy: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    min_depth_mm: int = 250,
    max_depth_mm: int = 10000,
    sample_step: int = 1,
    mad_k: float = 4.0,
    return_points: bool = False,
):
    """
    Computes a robust 3D position of an object in CAMERA coordinates (meters),
    using per-pixel depth + intrinsics.

    :param xy: (N,2) int array of (x,y) pixels for one object
    :param depth_mm: (H,W) uint16 depth aligned to RGB, units = mm
    :param K: (3,3) intrinsics for RGB at the SAME resolution as depth_mm
    :param min_depth_mm/max_depth_mm: keep consistent with your pipeline thresholdFilter
    :param sample_step: >1 to downsample object pixels for speed (keep 1 for full)
    :param mad_k: inlier gating aggressiveness (higher keeps more, lower rejects more)
    :param return_points: if True, also return the per-pixel 3D points used

    :return pos_m: (3,) float32 [X,Y,Z] in meters (camera frame)
    :return n_used: number of pixels used after filtering
    :return points_m: (M,3) float32 3D points used (optional)
    :rtype: (np.ndarray or None, int, np.ndarray)
    """
    if xy is None or xy.size == 0:
        return (None, 0, np.empty((0, 3), np.float32)) if return_points else (None, 0)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    pts = xy[::sample_step]
    xs = pts[:, 0].astype(np.int32)
    ys = pts[:, 1].astype(np.int32)

    h, w = depth_mm.shape[:2]
    inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[inb], ys[inb]
    if xs.size == 0:
        return (None, 0, np.empty((0, 3), np.float32)) if return_points else (None, 0)

    z = depth_mm[ys, xs].astype(np.float32)  # mm
    valid = (z > 0) & (z >= min_depth_mm) & (z <= max_depth_mm)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid]

    if z.size < 30:
        return (None, 0, np.empty((0, 3), np.float32)) if return_points else (None, 0)

    # Back-project to camera coordinates (meters)
    z_m = z / 1000.0
    x_m = (xs - cx) * z_m / fx
    y_m = (ys - cy) * z_m / fy

    P = np.column_stack((x_m, y_m, z_m)).astype(np.float32)  # (N,3)

    # ----------------------------
    # Robust inlier gating (MAD in each axis)
    # ----------------------------
    med = np.median(P, axis=0)
    sx = _mad(P[:, 0])
    sy = _mad(P[:, 1])
    sz = _mad(P[:, 2])

    # If MAD collapses (e.g., very flat distribution), keep it numerically stable
    sx = max(sx, 1e-6)
    sy = max(sy, 1e-6)
    sz = max(sz, 1e-6)

    dx = np.abs(P[:, 0] - med[0]) / sx
    dy = np.abs(P[:, 1] - med[1]) / sy
    dz = np.abs(P[:, 2] - med[2]) / sz

    inliers = (dx <= mad_k) & (dy <= mad_k) & (dz <= mad_k)
    P_in = P[inliers]

    # If gating is too strict (tiny object / noisy depth), fall back to ungated points
    if P_in.shape[0] < 30:
        P_in = P

    # ----------------------------
    # Final robust estimate: geometric median
    # ----------------------------
    pos = geometric_median(P_in, eps=1e-6, max_iter=128).astype(np.float32)

    if return_points:
        return pos, int(P_in.shape[0]), P_in
    return pos, int(P_in.shape[0])

def robust_positions_for_all_objects_camera_m(
    objects_xy: list,
    depth_mm: np.ndarray,
    K: np.ndarray,
    min_depth_mm: int = 250,
    max_depth_mm: int = 10000,
    sample_step: int = 1,
    mad_k: float = 4.0,
):
    """
    Computes robust (X,Y,Z) meters for each object in objects_xy.

    :param objects_xy: list of (N,2) int arrays of (x,y) pixels for each object
    :param depth_mm: (H,W) uint16 depth aligned to RGB, units = mm
    :param K: (3,3) intrinsics for RGB at the SAME resolution as depth_mm
    :param min_depth_mm: minimum depth to consider    
    :param max_depth_mm: maximum depth to consider
    :returns list of (pos_m or None, n_used)
    """
    out = []
    for xy in objects_xy:
        res = robust_object_position_camera_m(
            xy=xy,
            depth_mm=depth_mm,
            K=K,
            min_depth_mm=min_depth_mm,
            max_depth_mm=max_depth_mm,
            sample_step=sample_step,
            mad_k=mad_k,
            return_points=False,
        )
        pos_m = res[0]
        n_used = res[1]
        out.append((pos_m, n_used))
    return out
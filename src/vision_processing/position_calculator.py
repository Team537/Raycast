import numpy as np
import depthai as dai
import cv2

def draw_camera_facing_widget(
    frame_bgr: np.ndarray,
    q_wxyz: np.ndarray,
    R_imu_cam: np.ndarray | None = None,
    origin=(110, 110),
    radius=70,
):
    """
    Visualizes camera facing direction in a pitch-invariant way:
      1) build world<-imu rotation from quaternion
      2) apply IMU->camera extrinsics (optional)
      3) rotate camera forward axis into world
      4) project forward onto world XY plane and draw arrow
    """
    img = frame_bgr
    h, w = img.shape[:2]

    if R_imu_cam is None:
        R_imu_cam = np.eye(3, dtype=np.float32)

    # DepthAI rotation vector is referenced to gravity+north (NED), but we only need a consistent "world" frame here. :contentReference[oaicite:2]{index=2}
    R_world_imu = quat_to_R_wxyz(q_wxyz)

    # camera forward axis in camera coords: +Z forward (typical pinhole convention)
    f_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # world_forward = R_world_imu * R_imu_cam * f_cam
    world_forward = (R_world_imu @ (R_imu_cam @ f_cam)).astype(np.float32)

    a = heading_from_world_forward(world_forward)

    cx, cy = origin
    cx = int(np.clip(cx, radius + 5, w - radius - 5))
    cy = int(np.clip(cy, radius + 5, h - radius - 5))

    # ring + ticks
    cv2.circle(img, (cx, cy), radius, (255, 255, 255), 2)
    cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1)
    for deg in range(0, 360, 30):
        t = np.deg2rad(deg)
        x1 = int(cx + (radius - 8) * np.sin(t))
        y1 = int(cy - (radius - 8) * np.cos(t))
        x2 = int(cx + radius * np.sin(t))
        y2 = int(cy - radius * np.cos(t))
        cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 1)

    if a is None:
        cv2.putText(img, "heading: N/A (vertical)", (cx - radius, cy + radius + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return img

    # arrow (0 rad points up on widget)
    L = radius - 14
    x2 = int(cx + L * np.sin(a))
    y2 = int(cy - L * np.cos(a))
    cv2.arrowedLine(img, (cx, cy), (x2, y2), (0, 255, 0), 3, tipLength=0.25)  # :contentReference[oaicite:3]{index=3}
    cv2.putText(img, f"heading: {np.rad2deg(a):+.1f} deg", (cx - radius, cy + radius + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img

# ----------------------------
# Rotational calculation helpers
# ----------------------------
def heading_from_world_forward(world_forward: np.ndarray, eps: float = 1e-6) -> float | None:
    """
    Compute heading angle from a world-space forward vector by projecting onto horizontal plane.
    Assumes world Z is "up/down axis" (i.e., horizontal plane is XY).
    Returns radians, where 0 means "up" on the widget.
    """
    f = world_forward.astype(np.float32).copy()
    f[2] = 0.0  # project to horizontal plane
    n = float(np.linalg.norm(f))
    if n < eps:
        return None  # pointing too vertical; heading undefined
    f /= n
    # Map to widget: angle 0 points up (negative screen y), so use atan2(x, y)
    return float(np.arctan2(f[0], f[1]))

def quat_normalize_wxyz(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([1, 0, 0, 0], dtype=np.float32)

def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = map(float, a)
    bw, bx, by, bz = map(float, b)
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ], dtype=np.float32)

def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float32)

def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    # Standard 3-2-1 (yaw/pitch/roll) extraction for normalized quaternion :contentReference[oaicite:2]{index=2}
    w, x, y, z = map(float, q)
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    return float(np.arctan2(siny_cosp, cosy_cosp))

def quat_from_yaw_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)

class YawTare:
    def __init__(self):
        self._q_yaw0_inv = None  # inverse of initial yaw-only quat

    def apply(self, q_wxyz: np.ndarray) -> np.ndarray:
        q = quat_normalize_wxyz(q_wxyz)

        if self._q_yaw0_inv is None:
            yaw0 = yaw_from_quat_wxyz(q)
            q_yaw0 = quat_from_yaw_wxyz(yaw0)
            self._q_yaw0_inv = quat_conj_wxyz(quat_normalize_wxyz(q_yaw0))  # inverse

        # q_tared = inv(q_yaw0) ⊗ q
        q_tared = quat_mul_wxyz(self._q_yaw0_inv, q)
        return quat_normalize_wxyz(q_tared)

def remove_yaw_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q = quat_normalize_wxyz(q_wxyz)
    yaw = yaw_from_quat_wxyz(q)
    q_yaw = quat_from_yaw_wxyz(yaw)
    q_no_yaw = quat_mul_wxyz(quat_conj_wxyz(quat_normalize_wxyz(q_yaw)), q)
    return quat_normalize_wxyz(q_no_yaw)

def quat_to_R_wxyz(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] -> rotation matrix."""
    q = quat_normalize_wxyz(q)  
    w, x, y, z = map(float, q)
    n = w*w + x*x + y*y + z*z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    return np.array([
        [1.0 - (yy + zz),       xy - wz,       xz + wy],
        [      xy + wz, 1.0 - (xx + zz),       yz - wx],
        [      xz - wy,       yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float32)

# ----------------------------
# Positional calculation helpers
# ----------------------------
def _mad(x: np.ndarray) -> float:
    """
    Median Absolute Deviation (MAD) of a 1D array.
    :param x: (N,) array-like
    :return: MAD value
    """
    med = np.median(x)
    return float(np.median(np.abs(x - med))) + 1e-12

def geometric_median(points: np.ndarray, eps: float = 1e-6, max_iter: int = 128) -> np.ndarray:
    """
    Weiszfeld geometric median (robust).

    :param points: (N,3) array of 3D points
    :param eps: convergence threshold
    :param max_iter: maximum number of iterations
    :return: (3,) geometric median point
    """
    y = np.median(points, axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(points - y, axis=1)
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
def robust_object_position_world_m(
    xy: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    q_wxyz: np.ndarray,
    R_cam_imu: np.ndarray | None = None,
    min_depth_mm: int = 250,
    max_depth_mm: int = 10000,
    sample_step: int = 1,
    mad_k: float = 4.0,
    return_points: bool = False,
    imu_quat_is_imu_from_world: bool = False,
):
    """
    Returns robust object position [X,Y,Z] in meters after rotating by IMU orientation.

    Conventions:
      - q_wxyz is DepthAI rotation-vector quaternion referenced to gravity+north.
      - If imu_quat_is_imu_from_world=False (default), we treat q as R_world_imu (imu->world).
      - If True, we treat q as R_imu_world (world->imu). (Flip if your axes move the wrong way.)

    :param xy: (N,2) int array of (x,y) pixel coordinates belonging to the object
    :param depth_mm: (H,W) uint16 depth aligned to RGB, units = mm
    :param K: (3,3) intrinsics for RGB at the SAME resolution as depth_mm
    :param q_wxyz: (4,) float quaternion [w,x,y,z] from DepthAI IMU
    :param R_cam_imu: (3,3) rotation from camera frame to IMU frame (optional)
    :param min_depth_mm: minimum depth to consider
    :param max_depth_mm: maximum depth to consider
    :param sample_step: subsampling step for pixels
    :param mad_k: MAD gating threshold
    :param return_points: if True, also return the inlier 3D points in world frame
    :param imu_quat_is_imu_from_world: if True, treat imu quaternion as imu->world rotation
    :return: (pos_world_m or None, n_used, P_world or None) \n
        - pos_world_m: (3,) float robust position in world frame (meters) or None if failed
        - n_used: int number of points used in final estimation
        - P_world: (M,3) float array of inlier 3D points in world frame (meters) or None
    """
    if xy is None or xy.size == 0:
        return (None, 0, np.empty((0, 3), np.float32)) if return_points else (None, 0)

    if R_cam_imu is None:
        R_cam_imu = np.eye(3, dtype=np.float32)

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

    # Camera-frame backprojection (meters)
    z_m = z / 1000.0
    x_m = (xs - cx) * z_m / fx
    y_m = (ys - cy) * z_m / fy
    P_cam = np.column_stack((x_m, y_m, z_m)).astype(np.float32)

    # Robust MAD gating (axis-wise)
    med = np.median(P_cam, axis=0)
    sx = max(_mad(P_cam[:, 0]), 1e-6)
    sy = max(_mad(P_cam[:, 1]), 1e-6)
    sz = max(_mad(P_cam[:, 2]), 1e-6)

    dx = np.abs(P_cam[:, 0] - med[0]) / sx
    dy = np.abs(P_cam[:, 1] - med[1]) / sy
    dz = np.abs(P_cam[:, 2] - med[2]) / sz
    inliers = (dx <= mad_k) & (dy <= mad_k) & (dz <= mad_k)
    P_in = P_cam[inliers]
    if P_in.shape[0] < 30:
        P_in = P_cam

    # Final robust point in camera frame
    pos_cam = geometric_median(P_in).astype(np.float32)

    # Build rotation from quaternion
    R = quat_to_R_wxyz(q_wxyz)
    if imu_quat_is_imu_from_world:
        R_world_imu = R.T  # invert
    else:
        R_world_imu = R

    # Camera->IMU->World
    # v_imu = R_imu_cam @ v_cam, where R_imu_cam = R_cam_imu.T
    R_imu_cam = R_cam_imu.T
    R_world_cam = R_world_imu @ R_imu_cam

    pos_world = (R_world_cam @ pos_cam).astype(np.float32)

    if return_points:
        P_world = (R_world_cam @ P_in.T).T.astype(np.float32)
        return pos_world, int(P_in.shape[0]), P_world

    return pos_world, int(P_in.shape[0])

def robust_positions_for_all_objects_world_m(
    objects_xy: list,
    depth_mm: np.ndarray,
    K: np.ndarray,
    q_wxyz: np.ndarray,
    R_cam_imu: np.ndarray | None = None,
    min_depth_mm: int = 250,
    max_depth_mm: int = 10000,
    sample_step: int = 1,
    mad_k: float = 4.0,
    return_points: bool = False,
    imu_quat_is_imu_from_world: bool = False,
):
    """
    Computes robust (X,Y,Z) meters for each object in objects_xy.

    :param objects_xy: list of (N,2) int arrays of (x,y) pixels for each object
    :param depth_mm: (H,W) uint16 depth aligned to RGB, units = mm
    :param K: (3,3) intrinsics for RGB at the SAME resolution as depth_mm
    :param q_wxyz: (4,) float IMU quaternion in wxyz format
    :param R_cam_imu: (3,3) float camera to IMU rotation matrix or None
    :param min_depth_mm: minimum depth to consider    
    :param max_depth_mm: maximum depth to consider
    :returns list of (pos_m or None, n_used)
    """
    out = []
    for xy in objects_xy:
        res = robust_object_position_world_m(
            xy,
            depth_mm,
            K,
            q_wxyz,
            R_cam_imu,
            min_depth_mm,
            max_depth_mm,
            sample_step,
            mad_k,
            return_points,
            imu_quat_is_imu_from_world,
        )
        pos_m = res[0]
        n_used = res[1]
        out.append((pos_m, n_used))
    return out
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# ------------------------------
# Kalman Filter Setup / Robot Tracking
# ------------------------------
def _create_constant_velocity_kalman_filter_3d(delta_time_s: float) -> KalmanFilter:
    """
    Create a constant-velocity Kalman Filter in 3D.

    State vector (6D):
        [x, y, z, vx, vy, vz]  (meters, meters/sec) in WORLD coordinates

    Measurement vector (3D):
        [x, y, z]             (meters) in WORLD coordinates
    """
    kf = KalmanFilter(dim_x=6, dim_z=3)

    # ----------------------------
    # State transition model (F)
    # ----------------------------
    # x_k = x_{k-1} + vx * dt
    # vx_k = vx_{k-1}
    # (same for y/z)
    F = np.eye(6, dtype=float)
    F[0, 3] = delta_time_s
    F[1, 4] = delta_time_s
    F[2, 5] = delta_time_s
    kf.F = F

    # ----------------------------
    # Measurement model (H)
    # ----------------------------
    # We directly measure position only.
    H = np.zeros((3, 6), dtype=float)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    kf.H = H

    return kf


@dataclass
class RobotTrack3D:
    """
    A single tracked robot, containing:
      - A unique ID
      - A Kalman filter holding position + velocity
      - Lifecycle counters (hits, age, and missed frames)
    """
    track_id: int
    kalman_filter: KalmanFilter

    # Number of frames this track has been updated by a matched detection.
    total_updates: int = 0

    # Number of frames since track was created.
    total_frames_alive: int = 0

    # Number of consecutive frames with no detection match.
    missed_frames: int = 0

    # AI Detected Team #
    team__number = 0
    
    def predict_next_state(self) -> None:
        """Advance the track state forward one time-step using the motion model."""
        self.kalman_filter.predict()
        self.total_frames_alive += 1
        self.missed_frames += 1

    def correct_with_measurement(self, measured_position_world_m: np.ndarray) -> None:
        """
        Correct the predicted state with a new 3D measurement in world coordinates.
        """
        z = measured_position_world_m.reshape(3, 1)
        self.kalman_filter.update(z)
        self.total_updates += 1
        self.missed_frames = 0

    @property
    def position_world_m(self) -> np.ndarray:
        """Current estimated position (meters) in world coordinates."""
        return self.kalman_filter.x[:3].reshape(3)

    @property
    def velocity_world_mps(self) -> np.ndarray:
        """Current estimated velocity (m/s) in world coordinates."""
        return self.kalman_filter.x[3:6].reshape(3)


class RobotTracker3D:
    """
    Multi-object tracker for opponent robots using tracking-by-detection.

    Core loop per frame:
      1) Predict all existing tracks forward by dt
      2) Associate detections to tracks (Hungarian assignment)
      3) Update matched tracks with detections
      4) Spawn new tracks for unmatched detections
      5) Age out old tracks that have been missed for too long
    """

    def __init__(
        self,
        *,
        # How many frames to keep a track alive without any detection match.
        max_missed_frames: int = 15,

        # Require at least this many updates before we consider the track "reliable".
        min_updates_to_confirm: int = 2,

        # Gating distance: maximum allowed match distance (meters) between a detection and
        # a predicted track position. Prevents nonsense associations.
        association_gate_distance_m: float = 1.25,

        # Process noise scale: higher means the tracker tolerates more acceleration/turning.
        process_noise_scale: float = 2.0,

        # Measurement noise (meters): higher means we trust detections less (noisy depth).
        measurement_noise_m: float = 0.10,
    ):
        self.max_missed_frames = max_missed_frames
        self.min_updates_to_confirm = min_updates_to_confirm
        self.association_gate_distance_m = association_gate_distance_m
        self.process_noise_scale = process_noise_scale
        self.measurement_noise_m = measurement_noise_m

        self._active_tracks: List[RobotTrack3D] = []
        self._next_track_id: int = 1

    def _create_new_track(self, initial_position_world_m: np.ndarray, delta_time_s: float) -> RobotTrack3D:
        """
        Spawn a new track starting at the given position. Velocity starts at ~0 and
        is learned as we receive more frames.
        """
        kf = _create_constant_velocity_kalman_filter_3d(delta_time_s)

        # Initial state: position = measurement, velocity = 0
        kf.x = np.zeros((6, 1), dtype=float)
        kf.x[0:3, 0] = initial_position_world_m

        # Initial covariance:
        # - position somewhat confident (0.25m^2 variance)
        # - velocity very uncertain (4.0 (m/s)^2 variance)
        kf.P = np.diag([0.25, 0.25, 0.25, 4.0, 4.0, 4.0]).astype(float)

        # Measurement noise covariance (R)
        kf.R = (self.measurement_noise_m ** 2) * np.eye(3, dtype=float)

        # Process noise covariance (Q)
        # This is a simple diagonal approximation that works well in practice.
        # Increasing process_noise_scale allows quicker velocity changes (acceleration).
        dt = max(delta_time_s, 1e-3)
        pos_noise_var = (0.5 * dt * dt * self.process_noise_scale) ** 2
        vel_noise_var = (dt * self.process_noise_scale) ** 2
        kf.Q = np.diag([pos_noise_var, pos_noise_var, pos_noise_var,
                        vel_noise_var, vel_noise_var, vel_noise_var]).astype(float)

        new_track = RobotTrack3D(
            track_id=self._next_track_id,
            kalman_filter=kf,
            total_updates=1,   
            total_frames_alive=1,
            missed_frames=0,
        )
        self._next_track_id += 1
        return new_track

    def update_tracks(
        self,
        detected_robot_positions_world_m: List[np.ndarray],
        delta_time_s: float,
    ) -> List[RobotTrack3D]:
        """
        Update the tracker with the latest frame detections.

        :param detected_robot_positions_world_m:
            List of (3,) arrays in WORLD coordinates (meters).
        :param delta_time_s:
            Time since last frame (seconds).

        :return:
            List of active tracks after update. Tracks include those that are
            temporarily "missing" but still within the max_missed_frames window.
        """
        dt = max(delta_time_s, 1e-3)

        # ----------------------------------------------------------
        # 1) Predict all current tracks forward
        # ----------------------------------------------------------
        for track in self._active_tracks:
            # Update the filter's dt inside F (state transition)
            track.kalman_filter.F[0, 3] = dt
            track.kalman_filter.F[1, 4] = dt
            track.kalman_filter.F[2, 5] = dt

            # Update process noise Q each frame (depends on dt)
            pos_noise_var = (0.5 * dt * dt * self.process_noise_scale) ** 2
            vel_noise_var = (dt * self.process_noise_scale) ** 2
            track.kalman_filter.Q = np.diag([pos_noise_var, pos_noise_var, pos_noise_var,
                                             vel_noise_var, vel_noise_var, vel_noise_var]).astype(float)

            track.predict_next_state()

        # If we have no tracks, initialize everything from detections.
        if len(self._active_tracks) == 0:
            self._active_tracks = [
                self._create_new_track(z, dt) for z in detected_robot_positions_world_m
            ]
            return self.get_visible_tracks()

        # If we have no detections, just age out tracks and return.
        if len(detected_robot_positions_world_m) == 0:
            self._remove_expired_tracks()
            return self.get_visible_tracks()

        # ----------------------------------------------------------
        # 2) Build association cost matrix
        # ----------------------------------------------------------
        num_tracks = len(self._active_tracks)
        num_detections = len(detected_robot_positions_world_m)

        association_cost = np.zeros((num_tracks, num_detections), dtype=float)

        for track_index, track in enumerate(self._active_tracks):
            predicted_position = track.position_world_m
            for detection_index, detection_position in enumerate(detected_robot_positions_world_m):
                # Cost = Euclidean distance in 3D (meters)
                association_cost[track_index, detection_index] = float(
                    np.linalg.norm(predicted_position - detection_position)
                )

        # ----------------------------------------------------------
        # 3) Hungarian assignment (min-cost matching)
        # ----------------------------------------------------------
        matched_track_indices, matched_detection_indices = linear_sum_assignment(association_cost)

        tracks_matched: set[int] = set()
        detections_matched: set[int] = set()

        # ----------------------------------------------------------
        # 4) Apply gating + update matched tracks
        # ----------------------------------------------------------
        for track_index, detection_index in zip(matched_track_indices, matched_detection_indices):
            match_distance_m = association_cost[track_index, detection_index]

            # Gate out matches that are too far away (likely wrong association).
            if match_distance_m <= self.association_gate_distance_m:
                self._active_tracks[track_index].correct_with_measurement(
                    detected_robot_positions_world_m[detection_index]
                )
                tracks_matched.add(track_index)
                detections_matched.add(detection_index)

        # ----------------------------------------------------------
        # 5) Unmatched detections create new tracks
        # ----------------------------------------------------------
        for detection_index in range(num_detections):
            if detection_index not in detections_matched:
                self._active_tracks.append(
                    self._create_new_track(detected_robot_positions_world_m[detection_index], dt)
                )

        # ----------------------------------------------------------
        # 6) Prune tracks that have been missed too long
        # ----------------------------------------------------------
        self._remove_expired_tracks()
        return self.get_visible_tracks()

    def _remove_expired_tracks(self) -> None:
        """Remove tracks that have exceeded the allowed missed-frame limit."""
        self._active_tracks = [
            track for track in self._active_tracks
            if track.missed_frames <= self.max_missed_frames
        ]

    def get_visible_tracks(self) -> List[RobotTrack3D]:
        """
        Return tracks that are either confirmed or very new.
        This avoids spamming "ghost" IDs from single-frame noise.
        """
        visible: List[RobotTrack3D] = []
        for track in self._active_tracks:
            is_confirmed = track.total_updates >= self.min_updates_to_confirm
            is_new = track.total_frames_alive <= self.min_updates_to_confirm
            if is_confirmed or is_new:
                visible.append(track)
        return visible
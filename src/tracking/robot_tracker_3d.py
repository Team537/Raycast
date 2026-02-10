from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from filterpy.kalman import KalmanFilter

from vision_processing.robot_classifier import (
    RobotColor,
    extract_robot_info,
)

# ------------------------------
# Universal Constants
# ------------------------------
TEAM_NUMBER_CONFIDENCE_THRESHOLD = 0.10
TEAM_NUMBER_EXPANDED_CONFIDENCE_THRESHOLD = 0.40

NUM_FRAMES_BETWEEN_PROPERTY_UPDATE = 15  # tune

BIG_COST = 1e9


@dataclass(frozen=True)
class RobotDetection3D:
    """
    One detection in the current frame.
    """
    pos_world_m: np.ndarray                 # shape (3,)
    mask: np.ndarray                        # HxW, float/bool/u8 ok
    bbox_xyxy: Tuple[float, float, float, float]  # (x1,y1,x2,y2)


# ------------------------------
# Kalman Filter Setup / Robot Tracking
# ------------------------------
def _create_constant_velocity_kalman_filter_3d(delta_time_s: float) -> KalmanFilter:
    kf = KalmanFilter(dim_x=6, dim_z=3)

    F = np.eye(6, dtype=float)
    F[0, 3] = delta_time_s
    F[1, 4] = delta_time_s
    F[2, 5] = delta_time_s
    kf.F = F

    H = np.zeros((3, 6), dtype=float)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 2] = 1.0
    kf.H = H

    return kf


@dataclass
class RobotTrack3D:
    track_id: int
    kalman_filter: KalmanFilter

    total_updates: int = 0
    total_frames_alive: int = 0

    missed_frames: int = 0
    frames_since_property_update: int = 0

    min_updates_to_confirm: int = 2
    property_update_frames: int = NUM_FRAMES_BETWEEN_PROPERTY_UPDATE

    team_number: int = -1
    team_number_confidence: float = 0.0
    robot_color: RobotColor | None = None

    updating_properties: bool = False

    def predict_next_state(self) -> None:
        self.kalman_filter.predict()
        self.total_frames_alive += 1
        self.missed_frames += 1
        self.frames_since_property_update += 1

    def correct_with_measurement(self, measured_position_world_m: np.ndarray) -> None:
        z = measured_position_world_m.reshape(3, 1)
        self.kalman_filter.update(z)
        self.total_updates += 1
        self.missed_frames = 0

    def evaluate_robot_properties(
        self,
        color_frame: np.ndarray,
        bumper_mask: np.ndarray,
        bbox_xyxy: Tuple[float, float, float, float],
    ) -> None:
        if self.updating_properties:
            return

        self.frames_since_property_update = 0
        self.updating_properties = True
        try:
            color, est_team_num, team_num_conf = extract_robot_info(color_frame, bumper_mask, bbox_xyxy)

            # Color is cheap and generally reliable -> always update
            self.robot_color = color

            # If OCR failed or no change, stop
            if est_team_num == -1 or est_team_num == self.team_number:
                return

            # If the new number seems like it contains the old number (partial read),
            # require higher confidence before overwriting.
            old_str = str(self.team_number) if self.team_number != -1 else ""
            new_str = str(est_team_num)

            confidence_margin = TEAM_NUMBER_CONFIDENCE_THRESHOLD
            if old_str and (old_str in new_str or new_str in old_str):
                confidence_margin = TEAM_NUMBER_EXPANDED_CONFIDENCE_THRESHOLD

            # Only update if confidence meaningfully improves (or if we had nothing)
            if self.team_number == -1 or (team_num_conf >= self.team_number_confidence + confidence_margin):
                self.team_number = est_team_num
                self.team_number_confidence = team_num_conf

        finally:
            self.updating_properties = False

    @property
    def can_revaluate_properties(self) -> bool:
        return (
            (self.missed_frames == 0)
            and (not self.updating_properties)
            and (
                self.frames_since_property_update > self.property_update_frames
                or (self.is_confirmed and self.robot_color is None)
                or (self.is_confirmed and self.team_number == -1)
            )
        )

    @property
    def position_world_m(self) -> np.ndarray:
        return self.kalman_filter.x[:3].reshape(3)

    @property
    def velocity_world_mps(self) -> np.ndarray:
        return self.kalman_filter.x[3:6].reshape(3)

    @property
    def is_confirmed(self) -> bool:
        return self.total_updates >= self.min_updates_to_confirm


class RobotTracker3D:
    def __init__(
        self,
        *,
        max_missed_frames: int = 15,
        min_updates_to_confirm: int = 2,

        # gating / association
        gate_probability: float = 0.99,  # “optimal” is scenario-dependent; tune this
        process_noise_scale: float = 2.0,
        measurement_noise_m: float = 0.10,
        spawn_suppression_distance_m: float = 0.35,

        # optional ID-locking behavior (recommended once team numbers are known)
        team_mismatch_penalty_d2: float = 50.0,
    ):
        self.max_missed_frames = max_missed_frames
        self.min_updates_to_confirm = min_updates_to_confirm

        self.gate_probability = float(gate_probability)
        self._chi2_gate_d2 = float(chi2.ppf(self.gate_probability, df=3))  # dof=3 for (x,y,z)

        self.process_noise_scale = process_noise_scale
        self.measurement_noise_m = measurement_noise_m
        self.spawn_suppression_distance_m = spawn_suppression_distance_m

        self.team_mismatch_penalty_d2 = team_mismatch_penalty_d2

        self._active_tracks: List[RobotTrack3D] = []
        self._next_track_id: int = 1

    def _create_new_track(self, initial_position_world_m: np.ndarray, delta_time_s: float) -> RobotTrack3D:
        kf = _create_constant_velocity_kalman_filter_3d(delta_time_s)

        kf.x = np.zeros((6, 1), dtype=float)
        kf.x[0:3, 0] = initial_position_world_m

        kf.P = np.diag([0.25, 0.25, 0.25, 4.0, 4.0, 4.0]).astype(float)
        kf.R = (self.measurement_noise_m ** 2) * np.eye(3, dtype=float)

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
            min_updates_to_confirm=self.min_updates_to_confirm,
        )
        self._next_track_id += 1
        return new_track

    def update_tracks(
        self,
        detections: List[RobotDetection3D],
        delta_time_s: float,
        *,
        color_frame: Optional[np.ndarray] = None,
    ) -> List[RobotTrack3D]:
        """
        Update tracker using detections (3D pos + mask + bbox).

        If color_frame is provided, the tracker will run OCR/color extractor
        for eligible matched tracks (rate-limited by can_revaluate_properties).
        """
        dt = max(delta_time_s, 1e-3)

        # 1) Predict
        for track in self._active_tracks:
            track.kalman_filter.F[0, 3] = dt
            track.kalman_filter.F[1, 4] = dt
            track.kalman_filter.F[2, 5] = dt

            pos_noise_var = (0.5 * dt * dt * self.process_noise_scale) ** 2
            vel_noise_var = (dt * self.process_noise_scale) ** 2
            track.kalman_filter.Q = np.diag([pos_noise_var, pos_noise_var, pos_noise_var,
                                             vel_noise_var, vel_noise_var, vel_noise_var]).astype(float)

            track.predict_next_state()

        # No tracks yet -> spawn from detections
        if not self._active_tracks:
            self._active_tracks = [self._create_new_track(d.pos_world_m, dt) for d in detections]
            return self.get_visible_tracks()

        # No detections -> age out
        if not detections:
            self._remove_expired_tracks()
            return self.get_visible_tracks()

        # 2) Build association cost matrix (Mahalanobis d^2 with chi-square gate)
        num_tracks = len(self._active_tracks)
        num_dets = len(detections)

        cost = np.full((num_tracks, num_dets), BIG_COST, dtype=float)

        for ti, track in enumerate(self._active_tracks):
            kf = track.kalman_filter
            x = kf.x
            H = kf.H
            P = kf.P
            R = kf.R

            S = (H @ P @ H.T) + R  # innovation covariance

            # We’ll use solve(S, y) instead of inv(S) for stability
            for di, det in enumerate(detections):
                z = det.pos_world_m.reshape(3, 1)
                y = z - (H @ x)

                v = np.linalg.solve(S, y)
                d2 = float((y.T @ v).item())

                # Chi-square gate (dof=3)
                if d2 <= self._chi2_gate_d2:
                    # Optional: once a track has a confident team number, discourage mismatch switches
                    if track.team_number != -1 and track.team_number_confidence >= 0.25:
                        # we don't know det team_number unless we OCR it; so we only “lock”
                        # if we *already* have a number and it’s stable. You could extend this
                        # by caching OCR per-detection if you want.
                        pass

                    cost[ti, di] = d2

        # 3) Hungarian assignment
        matched_t, matched_d = linear_sum_assignment(cost)

        tracks_matched: set[int] = set()
        dets_matched: set[int] = set()

        # Map track index -> detection index for property updates
        match_map: Dict[int, int] = {}

        # 4) Apply matches (skip gated-out BIG_COST pairs)
        for ti, di in zip(matched_t, matched_d):
            d2 = cost[ti, di]
            if d2 >= BIG_COST:
                continue

            self._active_tracks[ti].correct_with_measurement(detections[di].pos_world_m)
            tracks_matched.add(ti)
            dets_matched.add(di)
            match_map[ti] = di

        # 4b) Update properties (OCR/color) for eligible *matched* tracks
        if color_frame is not None:
            for ti, di in match_map.items():
                t = self._active_tracks[ti]
                if t.can_revaluate_properties:
                    det = detections[di]
                    t.evaluate_robot_properties(color_frame, det.mask, det.bbox_xyxy)

        # 5) Spawn new tracks for unmatched detections (with suppression)
        for di in range(num_dets):
            if di in dets_matched:
                continue

            z = detections[di].pos_world_m
            too_close = any(np.linalg.norm(t.position_world_m - z) <= self.spawn_suppression_distance_m
                            for t in self._active_tracks)
            if too_close:
                continue

            self._active_tracks.append(self._create_new_track(z, dt))

        # 6) Prune
        self._remove_expired_tracks()
        return self.get_visible_tracks()

    def _remove_expired_tracks(self) -> None:
        pruned: list[RobotTrack3D] = []
        for t in self._active_tracks:
            if t.is_confirmed:
                if t.missed_frames <= self.max_missed_frames:
                    pruned.append(t)
            else:
                if t.missed_frames <= self.min_updates_to_confirm:
                    pruned.append(t)
        self._active_tracks = pruned

    def get_visible_tracks(self) -> List[RobotTrack3D]:
        visible: List[RobotTrack3D] = []
        for track in self._active_tracks:
            is_confirmed = track.total_updates >= self.min_updates_to_confirm
            is_new = track.total_frames_alive <= self.min_updates_to_confirm
            if is_confirmed or is_new:
                visible.append(track)
        return visible

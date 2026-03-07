# UDPSender.py
from __future__ import annotations

import json
import socket
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Optional
from tracking.robot_tracker_3d import RobotTrack3D

@dataclass(frozen=True)
class RobotDetectionWire:
    x: float
    y: float
    z: float
    teamNumber: int
    team_number_confidence: float
    allianceColor: Optional[str]
    radius: float = 0.0


# ------------------------------------------------------------
# UDP Sender
# ------------------------------------------------------------
class UDPRobotDetectionsSender:
    """
    Sends RobotDetectionWire[] over UDP as a single JSON array.
    """

    def __init__(
        self,
        *,
        target_ip: str,
        target_port: int,
        max_payload_bytes: int = 3500,
        socket_timeout_s: float = 0.0,
        debug: bool = False,
    ) -> None:
        self.target_ip = str(target_ip)
        self.target_port = int(target_port)
        self.max_payload_bytes = int(max_payload_bytes)
        self.debug = bool(debug)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if socket_timeout_s and socket_timeout_s > 0:
            self._sock.settimeout(float(socket_timeout_s))

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


    # -------------------------
    # Public API
    # -------------------------
    def send_tracks(self, tracks: Iterable[RobotTrack3D]) -> None:
        """
        Convert the given RobotTrack3D list into RobotDetectionWire[] and send.
        Each `track` is expected to have:
          - position_world_m -> array-like [x, y, z]
          - team_number -> int
          - robot_color -> enum or None (we map to "RED"/"BLUE"/None)

        If the resulting JSON is too large, we truncate the list until it fits.
        """
        detections = [self._track_to_wire(t) for t in tracks]
        payload = self._encode_detections_with_truncation(detections)

        if payload is None:
            # Nothing valid to send
            return

        try:
            self._sock.sendto(payload, (self.target_ip, self.target_port))
        except OSError as e:
            if self.debug:
                print(f"[UDPRobotDetectionsSender] sendto failed: {e}")


    # -------------------------
    # Internals
    # -------------------------
    def _track_to_wire(self, track: RobotTrack3D) -> RobotDetectionWire:
        pos = getattr(track, "position_world_m", None)
        if pos is None:
            # Fallback to zeros if something is off; better than crashing the vision loop
            x, y, z = 0.0, 0.0, 0.0
        else:
            # Round to reduce payload size and noise (helps stay under 4096)
            x = float(pos[0])
            y = float(pos[1])
            z = float(pos[2])

        team_number = int(getattr(track, "team_number", -1))
        team_number_confidence = float(getattr(track, "team_number_confidence", 0))
        alliance_color = self._map_color(getattr(track, "robot_color", None))
        radius = getattr(track, "radius_m", 0)

        return RobotDetectionWire(
            x=round(x, 3),
            y=round(y, 3),
            z=round(z, 3),
            team_number_confidence = team_number_confidence,
            teamNumber=team_number,
            allianceColor=alliance_color,
            radius=round(radius, 3),
        )

    @staticmethod
    def _map_color(robot_color: Any) -> Optional[str]:
        """
        Converts Python enum/value into something Gson can parse for `AllianceColor`.
        """
        if robot_color is None:
            return None

        # Common cases: Enum with .name, or string already
        name = getattr(robot_color, "name", None)
        if isinstance(name, str):
            upper = name.upper()
        elif isinstance(robot_color, str):
            upper = robot_color.upper()
        else:
            upper = str(robot_color).upper()

        if "RED" in upper:
            return "RED"
        if "BLUE" in upper:
            return "BLUE"
        return None

    def _encode_detections_with_truncation(self, detections: list[RobotDetectionWire]) -> Optional[bytes]:
        """
        Encode as JSON array with no whitespace. If too large, truncate from the end until it fits.
        """
        if not detections:
            # Send empty list (valid JSON, tiny)
            return b"[]"

        # Convert to plain dicts first (faster than repeatedly encoding dataclasses inside loop)
        items = [asdict(d) for d in detections]

        # Fast path: try full payload once
        payload = self._encode_items(items)
        if payload is not None:
            return payload

        # Truncate until it fits (keep earliest items; feel free to reverse if you prefer newest)
        lo = 0
        hi = len(items)

        # Binary search for the largest prefix that fits
        best: Optional[bytes] = None
        while lo <= hi:
            mid = (lo + hi) // 2
            test_items = items[:mid]
            test_payload = self._encode_items(test_items)
            if test_payload is not None:
                best = test_payload
                lo = mid + 1
            else:
                hi = mid - 1

        if best is None:
            # Even one element didn't fit (unlikely unless buffer is extremely small)
            if self.debug:
                print("[UDPRobotDetectionsSender] Could not fit any detections into one UDP packet.")
            return b"[]"

        if self.debug and len(best) < len(payload or b""):
            print(
                f"[UDPRobotDetectionsSender] Truncated detections: "
                f"{len(detections)} -> {json.loads(best.decode('utf-8')).__len__()} to fit payload"
            )
        return best

    def _encode_items(self, items: list[dict]) -> Optional[bytes]:
        # separators=(',', ':') removes spaces; ensure_ascii=False keeps it compact too
        data = json.dumps(items, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(data) <= self.max_payload_bytes:
            return data
        return None
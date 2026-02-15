import json
import socket
import threading
from typing import Any, Callable, Optional
import numpy as np

class UDPReceiver:
    """
    UDPReceiver listens for JSON messages over UDP.

    Each UDP datagram is expected to contain exactly one JSON message:
      {"robot_pose": {"translation": {"x":..,"y":..,"z":..}, "yaw_rad": ..},
      {"capture": {"inputFrame": ..., "outputFrame": ..., "depthFrame": ...}}
    """

    def __init__(
        self,
        update_robot_pose: Callable[[Any, Any], None],
        save_frames: Callable[[Any, Any, Any], None],
        ip: str = "0.0.0.0",
        port: int = 5801,
        *,
        recv_timeout_s: float = 0.2,
        max_packet_bytes: int = 2048,
        debug: bool = False,
    ) -> None:
        """
        Initialize the UDPReceiver.

        :param update_robot_pose: Callback to update robot pose.
        :param save_frames: Callback to save frames.
        :param ip: IP address to bind the server to.
        :param port: Port number to bind the server to.
        :param recv_timeout_s: Socket timeout for receiving data.
        :param max_packet_bytes: Maximum size of incoming UDP packets.
        :param debug: Whether to enable debug logging.
        """
        self.update_robot_pose = update_robot_pose
        self.save_frames = save_frames

        self.ip = ip
        self.port = port
        self.recv_timeout_s = recv_timeout_s
        self.max_packet_bytes = max_packet_bytes
        self.debug = debug

        self._running = False
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """
        Start the UDPReceiver to listen for incoming messages.
        """
        if self._running:
            return

        self._running = True
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.ip, self.port))
        self._sock.settimeout(self.recv_timeout_s)

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="UDPReceiverThread",
        )
        self._thread.start()

        if self.debug:
            print(f"[UDPReceiver] Listening on {self.ip}:{self.port}")

    def stop(self) -> None:
        """
        Stop the UDPReceiver and clean up resources.
        """
        self._running = False

        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self.debug:
            print("[UDPReceiver] Stopped")

    def _loop(self) -> None:
        assert self._sock is not None
        sock = self._sock

        while self._running:
            try:
                data, addr = sock.recvfrom(self.max_packet_bytes)
            except socket.timeout:
                continue
            except OSError:
                break

            if not data:
                continue

            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception as e:
                if self.debug:
                    preview = data[:200]
                    print(f"[UDPReceiver] Invalid JSON from {addr}: {e} | {preview!r}")
                continue

            if self.debug:
                print(f"[UDPReceiver] Packet from {addr} ({len(data)} bytes)")

            self._process_message(msg)

    def _process_message(self, parsed: Any) -> None:
        """
        Supports either a dict (single message) or a list of dicts (batch).
        """
        messages = parsed if isinstance(parsed, list) else [parsed]

        for item in messages:
            if not isinstance(item, dict):
                continue

            # 1) capture
            capture = item.get("capture")
            if isinstance(capture, dict):
                input_frame = capture.get("inputFrame")
                output_frame = capture.get("outputFrame")
                depth_frame = capture.get("depthFrame")
                try:
                    self.save_frames(input_frame, output_frame, depth_frame)
                except Exception as e:
                    if self.debug:
                        print(f"[UDPReceiver] save_frames error: {e}")

            # 2) robot_pose
            pose = item.get("robot_pose")
            if isinstance(pose, dict):
                try:
                    translation = pose["translation"]
                    translation_tuple = (translation["x", translation["y"]], translation["z"])
                    yaw = pose["yaw_rad"]
                    self.update_robot_pose(translation_tuple, yaw)
                except Exception as e:
                    if self.debug:
                        print(f"[UDPReceiver] update_robot_pose error: {e}")
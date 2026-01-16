import socket
import threading
import json
from typing import Callable, Optional, Any


class TCPReceiver:
    """
    TCPReceiver listens for newline-delimited JSON (NDJSON) over TCP.

    Expected message shape (examples):
      {"robot_pose": {...}}
      {"capture": {"inputFrame": ..., "outputFrame": ..., "depthFrame": ...}}
      or a list of such objects: [{"robot_pose": {...}}, {"capture": {...}}]

    Wiring:
      - update_robot_pose(pose_dict)
      - save_frames(input_frame, output_frame, depth_frame)
    """

    def __init__(
        self,
        update_robot_pose: Callable[[Any], None],
        save_frames: Callable[[Any, Any, Any], None],
        ip: str = "0.0.0.0",
        port: int = 5801,
        *,
        accept_timeout_s: float = 1.0,
        client_timeout_s: float = 2.0,
        debug: bool = False,
    ):
        self.update_robot_pose = update_robot_pose
        self.save_frames = save_frames

        self.ip = ip
        self.port = port
        self.accept_timeout_s = accept_timeout_s
        self.client_timeout_s = client_timeout_s
        self.debug = debug

        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._server_thread: Optional[threading.Thread] = None

        self._client_threads: list[threading.Thread] = []

    def start(self) -> None:
        """Starts the TCP server thread."""
        if self._running:
            return

        self._running = True
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.ip, self.port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(self.accept_timeout_s)

        self._server_thread = threading.Thread(
            target=self._server_loop,
            daemon=True,
            name="TCPReceiverServerThread",
        )
        self._server_thread.start()

        if self.debug:
            print(f"[TCPReceiver] Started on {self.ip}:{self.port}")

    def stop(self) -> None:
        """Stops the server and closes sockets to unblock threads."""
        self._running = False

        # Closing the server socket will unblock accept()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

        # Join server thread
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None

        # Best-effort join client threads
        for t in list(self._client_threads):
            if t.is_alive():
                t.join(timeout=1.0)

        self._client_threads.clear()

        if self.debug:
            print("[TCPReceiver] Stopped")

    def _server_loop(self) -> None:
        """Accepts connections and spawns a per-client handler."""
        assert self._server_socket is not None
        srv = self._server_socket

        while self._running:
            try:
                client_socket, client_address = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                # Socket closed during stop()
                break

            if self.debug:
                print(f"[TCPReceiver] Connection from {client_address}")

            t = threading.Thread(
                target=self._handle_client,
                args=(client_socket, client_address),
                daemon=True,
                name=f"TCPReceiverClientThread-{client_address[0]}:{client_address[1]}",
            )
            self._client_threads.append(t)
            t.start()

    def _handle_client(self, client_socket: socket.socket, client_address) -> None:
        """
        Reads NDJSON lines from the client and processes each JSON object.
        Handles partial TCP reads by buffering and splitting on '\n'.
        """
        with client_socket:
            client_socket.settimeout(self.client_timeout_s)
            buffer = b""

            while self._running:
                try:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        if self.debug:
                            print(f"[TCPReceiver] Client disconnected: {client_address}")
                        break

                    buffer += chunk

                    # Process full lines
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            msg = json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            if self.debug:
                                preview = line[:200]
                                print(f"[TCPReceiver] Invalid JSON from {client_address}: {preview!r}")
                            continue

                        self._process_message(msg)

                except socket.timeout:
                    # Keep looping so stop() can end us quickly
                    continue
                except OSError:
                    break
                except Exception as e:
                    if self.debug:
                        print(f"[TCPReceiver] Error with {client_address}: {e}")
                    break

    def _process_message(self, parsed: Any) -> None:
        """
        Supports either a dict (single message) or a list of dicts (batch).
        """
        # Normalize to list of dict-like items
        messages = parsed if isinstance(parsed, list) else [parsed]

        for item in messages:
            if not isinstance(item, dict):
                continue

            # 1) capture
            capture = item.get("capture")
            if capture is not None and isinstance(capture, dict):
                input_frame = capture.get("inputFrame")
                output_frame = capture.get("outputFrame")
                depth_frame = capture.get("depthFrame")

                try:
                    self.save_frames(input_frame, output_frame, depth_frame)
                except Exception as e:
                    if self.debug:
                        print(f"[TCPReceiver] save_frames error: {e}")

            # 2) robot_pose
            pose = item.get("robot_pose")
            if pose is not None:
                try:
                    self.update_robot_pose(pose)
                except Exception as e:
                    if self.debug:
                        print(f"[TCPReceiver] update_robot_pose error: {e}")
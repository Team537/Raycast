import socket
import json
import time
import threading
from typing import Optional


class TimeSyncServer:
    """
    UDP time sync server for RoboRIO.

    For each request it captures:
      - t2: epoch time (ns) when request was received
      - t3: epoch time (ns) immediately before response send

    Response JSON: {"t2": <int>, "t3": <int>, "seq": <int optional>}
    """

    def __init__(self, ip: str = "0.0.0.0", port: int = 6000, *, timeout_s: float = 1.0, debug: bool = False):
        self.ip = ip
        self.port = port
        self.timeout_s = timeout_s
        self.debug = debug

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="TimeSyncServerThread")
        self._thread.start()
        if self.debug:
            print(f"[TimeSyncServer] Started on {self.ip}:{self.port}")

    def stop(self) -> None:
        self._running = False

        # Closing the socket unblocks recvfrom immediately (faster than waiting for timeout).
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.debug:
            print("[TimeSyncServer] Stopped")

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock = sock

        # Allow quick restart without "address already in use" on some systems.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        sock.bind((self.ip, self.port))
        sock.settimeout(self.timeout_s)

        # Optional: reduce log spam; track count
        handled = 0

        try:
            while self._running:
                try:
                    data, addr = sock.recvfrom(2048)
                    # T2: time request was received (epoch ns)
                    t2 = time.time_ns()

                    # Parse request (optional)
                    seq = None
                    try:
                        # Your Java currently sends plain "TIME_SYNC". Keep compatibility:
                        if data.startswith(b"{"):
                            req = json.loads(data.decode("utf-8"))
                            seq = req.get("seq")
                        # else: treat as legacy "TIME_SYNC"
                    except Exception:
                        # If request is malformed, still respond (or you can ignore)
                        seq = None

                    # T3: immediately before sending response (epoch ns)
                    t3 = time.time_ns()

                    resp = {"t2": t2, "t3": t3}
                    if seq is not None:
                        resp["seq"] = seq

                    sock.sendto(json.dumps(resp).encode("utf-8"), addr)

                    handled += 1
                    if self.debug and handled % 50 == 0:
                        print(f"[TimeSyncServer] handled={handled} last_addr={addr}")

                except socket.timeout:
                    continue
                except OSError:
                    # Socket closed during stop()
                    break
                except Exception as e:
                    if self.debug:
                        print(f"[TimeSyncServer] Error: {e}")

        finally:
            try:
                sock.close()
            except OSError:
                pass
            self._sock = None

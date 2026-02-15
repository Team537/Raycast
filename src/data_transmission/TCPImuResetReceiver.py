import json
import socket
import threading
import time
from typing import Any, Callable, Optional


class TCPImuResetReceiver:
    """
    TCPImuResetReceiver listens for newline-delimited JSON commands over TCP and
    invokes a supplied callback when an IMU reset command is received.

    Expected message format (one JSON object per line):
      {"cmd":"zero_imu","cmd_id":12345}

    The receiver deduplicates command identifiers for a bounded time window to
    prevent repeated execution of the same command.
    """

    def __init__(
        self,
        zero_imu: Callable[[], None],
        ip: str = "0.0.0.0",
        port: int = 5802,
        *,
        accept_timeout_s: float = 1.0,
        client_timeout_s: float = 2.0,
        dedupe_ttl_s: float = 2.0,
        debug: bool = False,
    ) -> None:
        """
        Initializes the reset receiver.

        Args:
            zero_imu: Callback invoked when a valid, non-duplicate reset command is received.
            ip: Local interface address to bind.
            port: Local TCP port to bind.
            accept_timeout_s: Timeout for accept() to allow responsive shutdown.
            client_timeout_s: Timeout for recv() to allow responsive shutdown.
            dedupe_ttl_s: Time window for deduplicating cmd_id values.
            debug: Enables diagnostic logging.
        """
        self.zero_imu = zero_imu

        self.ip = ip
        self.port = port
        self.accept_timeout_s = accept_timeout_s
        self.client_timeout_s = client_timeout_s
        self.dedupe_ttl_s = dedupe_ttl_s
        self.debug = debug

        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._server_thread: Optional[threading.Thread] = None
        self._client_threads: list[threading.Thread] = []

        self._seen_cmd_ids: dict[int, float] = {}
        self._seen_lock = threading.Lock()

    def start(self) -> None:
        """
        Starts the TCP server and begins accepting client connections in a background thread.
        """
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
            name="TCPImuResetReceiverServerThread",
        )
        self._server_thread.start()

        if self.debug:
            print(f"[TCPImuResetReceiver] Listening on {self.ip}:{self.port}")

    def stop(self) -> None:
        """
        Stops the server and closes sockets to unblock blocking operations.
        """
        self._running = False

        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None

        for t in list(self._client_threads):
            if t.is_alive():
                t.join(timeout=1.0)
        self._client_threads.clear()

        if self.debug:
            print("[TCPImuResetReceiver] Stopped")

    def _server_loop(self) -> None:
        """
        Accepts new TCP connections and spawns a thread for each client.
        """
        assert self._server_socket is not None
        srv = self._server_socket

        while self._running:
            try:
                client_socket, client_address = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            if self.debug:
                print(f"[TCPImuResetReceiver] Client connected: {client_address}")

            t = threading.Thread(
                target=self._handle_client,
                args=(client_socket, client_address),
                daemon=True,
                name=f"TCPImuResetReceiverClientThread-{client_address[0]}:{client_address[1]}",
            )
            self._client_threads.append(t)
            t.start()

    def _handle_client(self, client_socket: socket.socket, client_address) -> None:
        """
        Receives NDJSON lines from a client, parses each JSON object, and processes it.
        """
        with client_socket:
            client_socket.settimeout(self.client_timeout_s)
            buffer = b""

            while self._running:
                try:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        if self.debug:
                            print(f"[TCPImuResetReceiver] Client disconnected: {client_address}")
                        break

                    buffer += chunk

                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            msg = json.loads(line.decode("utf-8"))
                        except json.JSONDecodeError:
                            if self.debug:
                                print(f"[TCPImuResetReceiver] Invalid JSON line: {line[:200]!r}")
                            continue

                        self._process_message(msg)

                except socket.timeout:
                    continue
                except OSError:
                    break
                except Exception as e:
                    if self.debug:
                        print(f"[TCPImuResetReceiver] Client error: {e}")
                    break

    def _process_message(self, msg: Any) -> None:
        """
        Validates a parsed JSON message and invokes the reset callback when appropriate.
        """
        if not isinstance(msg, dict):
            return

        cmd = msg.get("cmd")
        if cmd != "zero_imu":
            return

        cmd_id_raw = msg.get("cmd_id", None)
        if cmd_id_raw is None:
            if self.debug:
                print("[TCPImuResetReceiver] zero_imu received without cmd_id; ignored.")
            return

        try:
            cmd_id = int(cmd_id_raw)
        except Exception:
            if self.debug:
                print(f"[TCPImuResetReceiver] cmd_id is not an integer: {cmd_id_raw!r}")
            return

        if self._already_seen(cmd_id):
            if self.debug:
                print(f"[TCPImuResetReceiver] Duplicate cmd_id={cmd_id}; ignored.")
            return

        if self.debug:
            print(f"[TCPImuResetReceiver] Executing zero_imu cmd_id={cmd_id}")

        try:
            self.zero_imu()
        except Exception as e:
            if self.debug:
                print(f"[TCPImuResetReceiver] zero_imu callback error: {e}")

    def _already_seen(self, cmd_id: int) -> bool:
        """
        Records and deduplicates a command identifier.

        Returns:
            True if the cmd_id has been seen within the deduplication time window.
            False if it is new and has been recorded.
        """
        now = time.time()
        with self._seen_lock:
            expired = [k for k, ts in self._seen_cmd_ids.items() if now - ts > self.dedupe_ttl_s]
            for k in expired:
                self._seen_cmd_ids.pop(k, None)

            if cmd_id in self._seen_cmd_ids:
                return True

            self._seen_cmd_ids[cmd_id] = now
            return False

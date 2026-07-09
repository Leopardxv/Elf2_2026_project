"""
EEG Attention Monitor — reads BrainFlow focus metric via UDP.
Runs as a background thread, updates Context.attention continuously.
"""
import threading, time, socket, struct
from .context import ctx


class AttentionMonitor:
    """Background monitor: reads BrainFlow focus value from UDP port 12345."""

    def __init__(self, udp_port=12345):
        self._port = udp_port
        self._thread = None
        self._running = False
        self._latest = 0.5

    def start(self):
        """Start background monitoring thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Attention] Monitor started (UDP:{})".format(self._port))

    def stop(self):
        self._running = False

    @property
    def value(self) -> float:
        """Latest attention value (0=focused, 1=relaxed). Updated in real-time."""
        return self._latest

    def _loop(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(1.0)
            sock.bind(("0.0.0.0", self._port))
        except Exception as e:
            print("[Attention] UDP bind failed:", e)
            return

        while self._running:
            try:
                data, _ = sock.recvfrom(1024)
                # BrainFlow focus: 4-byte float
                if len(data) >= 4:
                    val = struct.unpack("f", data[:4])[0]
                    self._latest = max(0.0, min(1.0, val))
                    ctx.set_attention(self._latest)
            except socket.timeout:
                continue
            except Exception:
                continue
        sock.close()

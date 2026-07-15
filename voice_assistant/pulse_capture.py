"""Low-latency PulseAudio capture for the Bluetooth headset source."""
import os
import select
import subprocess
import time

import numpy as np


class PulseInput:
    """Read mono float32 frames from PulseAudio's current default source."""

    def __init__(self, sample_rate: int, block_frames: int):
        self._sample_rate = sample_rate
        self._block_frames = block_frames
        self._process = None
        self._pending = bytearray()

    def start(self):
        env = os.environ.copy()
        env.setdefault("PULSE_SERVER", f"unix:/run/user/{os.getuid()}/pulse/native")
        self._process = subprocess.Popen(
            [
                "parec", "--raw", "--format=s16le",
                f"--rate={self._sample_rate}", "--channels=1",
                "--latency-msec=40",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        time.sleep(0.05)
        if self._process.poll() is not None:
            error = self._process.stderr.read().decode("utf-8", "replace").strip()
            raise RuntimeError(error or "parec could not open the PulseAudio source")

    def read(self, timeout: float):
        if not self._process or not self._process.stdout:
            raise RuntimeError("PulseAudio capture is not running")

        wanted = self._block_frames * np.dtype(np.int16).itemsize
        deadline = time.monotonic() + timeout
        fd = self._process.stdout.fileno()
        while len(self._pending) < wanted:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                return None
            data = os.read(fd, wanted - len(self._pending))
            if not data:
                error = self._process.stderr.read().decode("utf-8", "replace").strip()
                raise RuntimeError(error or "PulseAudio capture stopped")
            self._pending.extend(data)

        raw = bytes(self._pending[:wanted])
        del self._pending[:wanted]
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    def close(self):
        process = self._process
        self._process = None
        if not process:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

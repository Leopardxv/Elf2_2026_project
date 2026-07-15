#!/usr/bin/env python3
"""
Audio I/O — lightweight wrapper around sounddevice for RK3588.

Hardware:
  Mic:  PulseAudio default source (BBH headset microphone)
  Spk:  PulseAudio default sink (BBH headset speaker)

All internal processing at 16000 Hz mono float32.
"""
import numpy as np
import sounddevice as sd
from scipy.signal import resample
from typing import Optional, Callable
import os

SAMPLE_RATE = 16000
CHANNELS = int(os.getenv("VOICE_MIC_CHANNELS", "1"))
DTYPE = np.float32

_MIC_DEVICE = os.getenv("VOICE_MIC_DEVICE", "default")


class AudioCapture:
    """Streaming microphone capture with callback."""

    def __init__(self, callback: Callable[[np.ndarray], None],
                 block_size: int = 1600):
        self._callback = callback
        self._block_size = block_size
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    def _audio_cb(self, indata: np.ndarray, frames, time, status):
        if status:
            print(f"[Audio] Warning: {status}")
        mono = indata[:, 0]
        self._callback(mono)

    def start(self):
        if self._running:
            return
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self._block_size,
            device=_MIC_DEVICE,
            callback=self._audio_cb,
        )
        self._stream.start()
        self._running = True

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running


class AudioRecorder:
    """Buffered microphone capture for utterance recording."""

    def __init__(self, block_size: int = 1600):
        self._block_size = block_size
        self._buffer: list = []
        self._stream: Optional[sd.InputStream] = None

    def _audio_cb(self, indata: np.ndarray, frames, time, status):
        self._buffer.append(indata[:, 0].copy())

    def start(self):
        self._buffer = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self._block_size,
            device=_MIC_DEVICE,
            callback=self._audio_cb,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._buffer:
            return np.array([], dtype=DTYPE)
        return np.concatenate(self._buffer)


class AudioPlayer:
    """Play audio through PulseAudio's BBH default sink."""

    def play(self, audio: np.ndarray, sample_rate: int = 44100):
        if audio.size == 0:
            return
        import tempfile, subprocess, wave
        audio_out = np.clip(audio.astype(np.float32).ravel(), -1.0, 1.0)
        peak = np.abs(audio_out).max()
        if peak > 0 and peak < 0.3:
            audio_out = audio_out * (0.95 / peak)
        audio_i16 = (np.clip(audio_out, -1.0, 1.0) * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            with wave.open(tmp, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(audio_i16.tobytes())
            duration = len(audio_out) / max(sample_rate, 1)
            timeout = max(3, int(duration) + 3)
            env = os.environ.copy()
            env["PULSE_LATENCY_MSEC"] = "10"
            subprocess.run(
                ["paplay", tmp],
                timeout=timeout, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            pass  # expected — PulseAudio DP hangs after buffer drain
        except Exception as e:
            print(f"[Audio] Playback error: {e}")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def play_blocking(self, audio: np.ndarray, sample_rate: int = 44100):
        self.play(audio, sample_rate)

#!/usr/bin/env python3
"""Command ASR using a Chinese Zipformer model after wake-up."""
import os
import time
from collections import deque

import numpy as np
import sherpa_onnx

from voice_assistant.pulse_capture import PulseInput

SAMPLE_RATE = 16000
ENERGY_THRESHOLD = float(os.getenv("VOICE_ASR_ENERGY_THRESHOLD", "0.008"))
MAX_ENERGY_THRESHOLD = float(os.getenv("VOICE_ASR_MAX_ENERGY_THRESHOLD", "0.04"))
SPEECH_START_TIMEOUT = 3.0

MODEL_DIR = os.getenv("VOICE_ASR_MODEL_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "asr", "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
))


class ASREngine:
    """Voice-command recognition with VAD and compact offline decoding."""

    def __init__(self, model_dir: str = None):
        self._model_dir = model_dir or MODEL_DIR
        self._recognizer = None

    def _model_file(self, *names: str) -> str:
        for name in names:
            path = os.path.join(self._model_dir, name)
            if os.path.isfile(path):
                return path
        return os.path.join(self._model_dir, names[0])

    def load(self) -> bool:
        paths = {
            "tokens": os.path.join(self._model_dir, "tokens.txt"),
            "encoder": self._model_file("encoder.int8.onnx", "encoder-epoch-99-avg-1.int8.onnx"),
            "decoder": self._model_file("decoder.onnx", "decoder-epoch-99-avg-1.onnx"),
            "joiner": self._model_file("joiner.int8.onnx", "joiner-epoch-99-avg-1.int8.onnx"),
        }
        missing = [name for name, path in paths.items() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"Zipformer ASR files missing: {', '.join(missing)}")
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=paths["tokens"],
            encoder=paths["encoder"],
            decoder=paths["decoder"],
            joiner=paths["joiner"],
            num_threads=2,
            sample_rate=SAMPLE_RATE,
            enable_endpoint_detection=False,
        )
        print(f"[ASR] Zipformer loaded: {self._model_dir}")
        return True

    def listen_and_transcribe(self, timeout: float = 15.0) -> str:
        if self._recognizer is None:
            raise RuntimeError("[ASR] Model not loaded.")
        print("[ASR] Listening...")
        return self._record_and_transcribe(
            timeout=timeout,
            speech_start_timeout=SPEECH_START_TIMEOUT,
            max_silence=1.1,
            min_speech=0.3,
        )

    def _decode(self, audio: np.ndarray) -> str:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        return self._recognizer.get_result(stream).strip()

    @staticmethod
    def _normalize_quiet_speech(audio: np.ndarray) -> tuple[np.ndarray, float]:
        """Lift quiet HFP speech without touching normal or loud recordings."""
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak <= 1e-4 or peak >= 0.35:
            return audio, 1.0
        gain = min(2.4, 0.42 / peak)
        return np.clip(audio * gain, -0.95, 0.95), gain

    @staticmethod
    def _bandpass_hfp(audio: np.ndarray) -> np.ndarray:
        """Apply a lightweight 90-3600 Hz band-pass for HFP recordings."""
        x = np.asarray(audio, dtype=np.float32)
        x = x - np.median(x)
        high = np.empty_like(x)
        low = np.empty_like(x)
        high_alpha = 0.966  # 90 Hz high-pass at 16 kHz
        low_alpha = 0.586   # 3.6 kHz low-pass at 16 kHz
        prev_x = float(x[0]) if x.size else 0.0
        prev_high = 0.0
        prev_low = 0.0
        for index, value in enumerate(x):
            current = float(value)
            prev_high = high_alpha * (prev_high + current - prev_x)
            prev_x = current
            prev_low += low_alpha * (prev_high - prev_low)
            high[index] = prev_high
            low[index] = prev_low
        return low

    @staticmethod
    def _apply_agc(audio: np.ndarray) -> tuple[np.ndarray, float]:
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        gain = min(3.0, max(1.0, 0.12 / max(rms, 1e-4)))
        return np.clip(audio * gain, -0.95, 0.95), gain

    def _record_and_transcribe(self, timeout: float, speech_start_timeout: float,
                               max_silence: float, min_speech: float) -> str:
        block_size = 1600
        pre_roll = deque(maxlen=3)
        recorded_blocks = []
        speech_blocks = 0
        silence_blocks = 0
        min_speech_blocks = max(1, int(min_speech * SAMPLE_RATE / block_size))
        max_silence_blocks = max(1, int(max_silence * SAMPLE_RATE / block_size))
        noise_samples = deque(maxlen=10)
        threshold = ENERGY_THRESHOLD
        speech_started = False
        recorded_samples = 0
        peak = 0.0

        def keep_block(block):
            nonlocal recorded_samples, peak
            recorded_blocks.append(block.copy())
            recorded_samples += len(block)
            peak = max(peak, float(np.max(np.abs(block))))

        capture = PulseInput(SAMPLE_RATE, block_size)
        capture.start()
        start_time = time.monotonic()
        try:
            while time.monotonic() - start_time < timeout:
                block = capture.read(timeout=0.3)
                if block is None:
                    continue
                if (not speech_started and
                        time.monotonic() - start_time >= speech_start_timeout):
                    break

                energy = float(np.sqrt(np.mean(block ** 2)))
                if not speech_started:
                    pre_roll.append(block)
                    noise_samples.append(energy)
                    noise_floor = float(np.median(noise_samples))
                    threshold = min(
                        MAX_ENERGY_THRESHOLD,
                        max(ENERGY_THRESHOLD, noise_floor * 1.6),
                    )
                    if energy <= threshold:
                        continue
                    speech_started = True
                    for buffered_block in pre_roll:
                        keep_block(buffered_block)
                    speech_blocks = 1
                    silence_blocks = 0
                    continue

                keep_block(block)
                if energy > threshold:
                    speech_blocks += 1
                    silence_blocks = 0
                else:
                    silence_blocks += 1
                if (speech_blocks >= min_speech_blocks and
                        silence_blocks >= max_silence_blocks):
                    break
        finally:
            capture.close()

        if speech_blocks < min_speech_blocks:
            print(f"[ASR] No speech (threshold={threshold:.3f})")
            return ""

        audio = np.concatenate(recorded_blocks).astype(np.float32, copy=False)
        audio, gain = self._normalize_quiet_speech(audio)
        decode_started = time.monotonic()
        text = self._decode(audio)
        print(
            f"[ASR] Recorded {recorded_samples / SAMPLE_RATE:.1f}s, "
            f"threshold={threshold:.3f}, peak={int(peak * 32767)}, gain={gain:.2f}"
        )
        print(f"[ASR] Zipformer decode: {time.monotonic() - decode_started:.2f}s")
        print(f"[ASR] Result: '{text}'")
        return text

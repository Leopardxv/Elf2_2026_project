#!/usr/bin/env python3
"""
ASR Engine — Speech recognition using vosk (offline, ARM-optimized).

Records from mic with energy-based VAD, then transcribes.
"""
import os
import json
import numpy as np
import vosk
import sounddevice as sd

SAMPLE_RATE = 16000
MIC_DEVICE = 0
CHANNELS = 2
MIC_CHANNEL = 1  # right channel = main mic on NAU8822

ENERGY_THRESHOLD = 0.03

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "vosk", "vosk-model-small-cn-0.22"
)


class ASREngine:
    """
    Record-then-transcribe speech recognition using vosk.

    Usage:
        asr = ASREngine(); asr.load()
        text = asr.listen_and_transcribe(timeout=15)
    """

    def __init__(self, model_path: str = None):
        self._model_path = model_path or MODEL_PATH
        self._model = None

    def load(self) -> bool:
        vosk.SetLogLevel(-1)
        self._model = vosk.Model(self._model_path)
        print(f"[ASR] Vosk model loaded: {self._model_path}")
        return True

    def listen_and_transcribe(self, timeout: float = 15.0) -> str:
        """Record from mic, then transcribe. Returns transcribed text."""
        if not self._model:
            raise RuntimeError("[ASR] Model not loaded.")

        print("[ASR] Listening...")
        audio_i16 = self._record(timeout=timeout, max_silence=0.8, min_speech=0.3)
        if audio_i16 is None or len(audio_i16) == 0:
            return ""

        return self._transcribe(audio_i16)

    def _record(self, timeout: float, max_silence: float,
                min_speech: float):
        """Record with energy-based VAD. Returns int16 mono array or None."""
        import time as _time
        block_size = 3200  # 200ms at 16kHz
        buffer: list = []
        silence_frames = 0
        max_silence_frames = int(max_silence * SAMPLE_RATE / block_size)
        speech_frames = 0
        min_speech_frames = int(min_speech * SAMPLE_RATE / block_size)

        def audio_cb(indata, frames, t, status):
            ch = indata[:, MIC_CHANNEL] if indata.shape[1] > 1 else indata[:, 0]
            buffer.append(ch.ravel().copy())

        istream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            blocksize=block_size,
            device=MIC_DEVICE,
            callback=audio_cb,
        )

        start_time = _time.time()
        istream.start()

        try:
            while _time.time() - start_time < timeout:
                _time.sleep(0.15)

                if len(buffer) < 2:
                    continue

                recent = buffer[-2:]
                energy = max(float(np.sqrt(np.mean(b ** 2))) for b in recent)

                if energy > ENERGY_THRESHOLD:
                    speech_frames += 1
                    silence_frames = 0
                else:
                    silence_frames += 1

                if speech_frames >= min_speech_frames and silence_frames >= max_silence_frames:
                    break

        finally:
            istream.stop()
            istream.close()

        if speech_frames < min_speech_frames:
            return None

        audio_f32 = np.concatenate(buffer) if buffer else np.array([], dtype=np.float32)
        audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
        print(f"[ASR] Recorded {len(audio_i16) / SAMPLE_RATE:.1f}s, peak={np.abs(audio_i16).max()}")
        return audio_i16

    def _transcribe(self, audio_i16: np.ndarray) -> str:
        """Run vosk on int16 audio, return text."""
        rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
        rec.SetWords(True)
        chunk_size = 4000
        for i in range(0, len(audio_i16), chunk_size):
            chunk = audio_i16[i:i + chunk_size].tobytes()
            rec.AcceptWaveform(chunk)
        result = json.loads(rec.FinalResult())
        text = result.get("text", "").strip()
        print(f"[ASR] Result: '{text}'")
        return text

    @property
    def loaded(self) -> bool:
        return self._model is not None

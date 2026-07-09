#!/usr/bin/env python3
"""
Wake Word Engine — Real-time wake word detection using vosk streaming.

Streaming approach: feeds audio to vosk recognizer in real-time,
checks PartialResult for wake keyword after every chunk.
Triggers immediately when detected — no need to wait for silence.
"""
import os
import json
import threading
import time
from collections import deque
import numpy as np
import vosk
import sounddevice as sd

SAMPLE_RATE = 16000
MIC_DEVICE = 0
CHANNELS = 2
MIC_CHANNEL = 1

WAKE_KEYWORDS = ["精灵", "经历", "经理", "清明", "机灵", "丁隐", "明明", "精品"]

VAD_CHUNK_SEC = 0.1
MAX_UTTERANCE_SEC = 8.0

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "vosk", "vosk-model-small-cn-0.22"
)


class WakeWordEngine:
    """
    Streaming wake word detection using vosk PartialResult.

    Usage:
        engine = WakeWordEngine(); engine.load()
        engine.start(on_wake=lambda: print("woken!"))
        engine.stop()
    """

    def __init__(self, model_path: str = None):
        self._model_path = model_path or MODEL_PATH
        self._model = None
        self._on_wake = None
        self._running = False
        self._thread = None

    def load(self) -> bool:
        vosk.SetLogLevel(-1)
        self._model = vosk.Model(self._model_path)
        print(f"[KWS] Vosk streaming model loaded")
        return True

    def start(self, on_wake: callable):
        if not self._model:
            raise RuntimeError("[KWS] Model not loaded.")
        self._on_wake = on_wake
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        print("[KWS] Listening for wake word...")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        print("[KWS] Stopped")

    def reset(self):
        pass

    def _stream_loop(self):
        chunk_frames = int(VAD_CHUNK_SEC * SAMPLE_RATE)
        max_chunks = int(MAX_UTTERANCE_SEC / VAD_CHUNK_SEC)

        rec = None
        buffer_chunks = 0
        silence_chunks = 0
        in_utterance = False
        last_text = ""
        detected = False

        def callback(indata, frames, t, status):
            nonlocal rec, buffer_chunks, silence_chunks, in_utterance, last_text, detected
            if not self._running:
                return

            ch = indata[:, MIC_CHANNEL] if indata.shape[1] > 1 else indata[:, 0]
            chunk_i16 = (np.clip(ch.ravel(), -1.0, 1.0) * 32767).astype(np.int16)
            energy = float(np.sqrt(np.mean(ch.ravel().astype(np.float32) ** 2)))
            is_speech = energy > 0.08

            if detected:
                return

            if is_speech:
                if not in_utterance:
                    rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE)
                    rec.SetWords(False)
                    rec.SetMaxAlternatives(0)
                    buffer_chunks = 0
                    silence_chunks = 0
                    in_utterance = True
                    last_text = ""
                silence_chunks = 0
            elif in_utterance:
                silence_chunks += 1

            if in_utterance:
                rec.AcceptWaveform(chunk_i16.tobytes())
                buffer_chunks += 1

                # Check partial result every chunk — immediate detection
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial and partial != last_text:
                    last_text = partial
                    if any(kw in partial for kw in WAKE_KEYWORDS):
                        print(f"[KWS] PARTIAL: '{partial}' → WAKE!")
                        detected = True
                        in_utterance = False; rec = None
                        if self._on_wake:
                            self._on_wake()
                        return

                # End utterance after 1.5s silence or max duration
                if silence_chunks >= 15 or buffer_chunks >= max_chunks:
                    final = json.loads(rec.FinalResult()).get("text", "")
                    if final:
                        print(f"[KWS] FINAL: '{final}'")
                        if any(kw in final for kw in WAKE_KEYWORDS):
                            print(f"[KWS] FINAL → WAKE!")
                            detected = True
                            if self._on_wake:
                                self._on_wake()
                    in_utterance = False
                    rec = None

        istream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            blocksize=chunk_frames,
            device=MIC_DEVICE,
            callback=callback,
        )
        istream.start()
        while self._running:
            time.sleep(0.05)
        istream.stop()
        istream.close()

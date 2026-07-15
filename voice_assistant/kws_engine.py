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
import numpy as np
import vosk
from voice_assistant.pulse_capture import PulseInput

SAMPLE_RATE = 16000
# ALSA's default device is PulseAudio, whose default source is enforced by
# setup_bluetooth.sh to be the BBH headset microphone.
MIC_DEVICE = os.getenv("VOICE_MIC_DEVICE", "default")
CHANNELS = int(os.getenv("VOICE_MIC_CHANNELS", "1"))
MIC_CHANNEL = 0

WAKE_KEYWORDS = ["精灵", "经历", "经理", "清明", "机灵", "丁隐", "明明", "精品"]
WAKE_PHRASE = "精灵精灵"
PRIMARY_WAKE_KEYWORD = "\u7cbe\u7075"

VAD_CHUNK_SEC = 0.1
MAX_UTTERANCE_SEC = 4.0
END_SILENCE_CHUNKS = 8
# BBH HFP speech measures around 0.014 RMS; 0.08 rejects normal speech.
SPEECH_ENERGY_THRESHOLD = float(
    os.getenv("VOICE_KWS_ENERGY_THRESHOLD", "0.008")
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "vosk", "vosk-model-small-cn-0.22"
)


def _is_wake_text(text: str) -> bool:
    normalized = "".join(text.split())
    if WAKE_PHRASE in normalized:
        return True
    tokens = text.replace("[unk]", " ").split()
    if PRIMARY_WAKE_KEYWORD in tokens:
        return True
    return sum(token in WAKE_KEYWORDS for token in tokens) >= 2


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
        self._state_lock = threading.Lock()

    def load(self) -> bool:
        vosk.SetLogLevel(-1)
        self._model = vosk.Model(self._model_path)
        print(f"[KWS] Vosk streaming model loaded")
        return True

    def share_model(self, model):
        self._model = model

    def start(self, on_wake: callable):
        if not self._model:
            raise RuntimeError("[KWS] Model not loaded.")
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                return
            self._on_wake = on_wake
            self._running = True
            self._thread = threading.Thread(target=self._stream_loop, daemon=True)
            self._thread.start()
        print("[KWS] Listening for wake word...")

    def stop(self):
        with self._state_lock:
            self._running = False
            thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self._state_lock:
            self._thread = thread if thread and thread.is_alive() else None
        print("[KWS] Stopped")

    def reset(self):
        pass

    def _stream_loop(self):
        chunk_frames = int(VAD_CHUNK_SEC * SAMPLE_RATE)
        max_chunks = int(MAX_UTTERANCE_SEC / VAD_CHUNK_SEC)
        grammar = json.dumps(
            WAKE_KEYWORDS + ["[unk]"],
            ensure_ascii=False,
        )
        rec = vosk.KaldiRecognizer(self._model, SAMPLE_RATE, grammar)
        rec.SetWords(False)
        rec.SetMaxAlternatives(0)
        buffer_chunks = 0
        silence_chunks = 0
        in_utterance = False
        last_text = ""
        detected = False

        capture = None
        while self._running and capture is None:
            try:
                capture = PulseInput(SAMPLE_RATE, chunk_frames)
                capture.start()
            except RuntimeError as exc:
                capture = None
                print(f"[KWS] Microphone unavailable, retrying: {exc}")
                time.sleep(1.0)

        if capture is None:
            return
        try:
            while self._running:
                try:
                    chunk_f32 = capture.read(timeout=0.1)
                except RuntimeError as exc:
                    print(f"[KWS] Microphone unavailable, retrying: {exc}")
                    return
                if chunk_f32 is None:
                    continue

                if detected:
                    continue

                energy = float(np.sqrt(np.mean(chunk_f32 * chunk_f32)))
                chunk_i16 = (
                    np.clip(chunk_f32, -1.0, 1.0) * 32767
                ).astype(np.int16)

                if energy > SPEECH_ENERGY_THRESHOLD:
                    if not in_utterance:
                        rec.Reset()
                        buffer_chunks = 0
                        silence_chunks = 0
                        in_utterance = True
                        last_text = ""
                    silence_chunks = 0
                elif in_utterance:
                    silence_chunks += 1

                if not in_utterance:
                    continue

                rec.AcceptWaveform(chunk_i16.tobytes())
                buffer_chunks += 1

                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial and partial != last_text:
                    last_text = partial
                    if _is_wake_text(partial):
                        print(f"[KWS] PARTIAL: '{partial}' → WAKE!")
                        detected = True
                        in_utterance = False
                        if self._on_wake:
                            self._on_wake()
                        continue

                if (silence_chunks >= END_SILENCE_CHUNKS or
                        buffer_chunks >= max_chunks):
                    final = json.loads(rec.FinalResult()).get("text", "")
                    if final:
                        print(f"[KWS] FINAL: '{final}'")
                        if _is_wake_text(final):
                            print("[KWS] FINAL → WAKE!")
                            detected = True
                            if self._on_wake:
                                self._on_wake()
                    in_utterance = False
                    rec.Reset()
        finally:
            capture.close()

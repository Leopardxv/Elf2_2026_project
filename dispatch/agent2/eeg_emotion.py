"""
EEG Emotion Monitor — runs EEG-Conformer on NPU periodically.
Updates Context.emotion in background.
"""
import threading, time, os, numpy as np
from .context import ctx


class EmotionMonitor:
    """Background monitor: runs EEG-Conformer on NPU every N seconds."""

    EMOTIONS = ["neutral", "positive", "negative"]

    def __init__(self, interval=5.0):
        self._interval = interval
        self._thread = None
        self._running = False
        self._rknn = None
        self._eeg_data = None
        self._latest_label = "neutral"
        self._latest_conf = 0.0

    def load(self) -> bool:
        from rknnlite.api import RKNNLite
        model = "/home/elf/Projects/emotions/EEG-Conformer/eeg_conformer.rknn"
        if not os.path.isfile(model):
            print("[Emotion] Model not found")
            return False
        self._rknn = RKNNLite()
        if self._rknn.load_rknn(model) != 0:
            print("[Emotion] load failed")
            return False
        if self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            print("[Emotion] init failed")
            return False

        # Load EEG data (REAL data only - no simulation fallback)
        eeg_path = "/home/elf/Projects/emotions/S1_session1.npy"
        self._eeg_data = None
        if os.path.isfile(eeg_path):
            try:
                data = np.load(eeg_path, allow_pickle=True)
                self._eeg_data = np.array(data, dtype=np.float32)
                print("[Emotion] EEG data loaded: {} samples".format(len(self._eeg_data)))
            except Exception as e:
                print("[Emotion] ERROR: Cannot load EEG data: {}".format(e))
                print("[Emotion] Emotion monitor will run in PASSIVE mode (no data)")
        else:
            print("[Emotion] WARNING: EEG data file not found: {}".format(eeg_path))
            print("[Emotion] Emotion monitor will run in PASSIVE mode - connect EEG hardware")

        print("[Emotion] EEG-Conformer loaded on NPU")
        return True

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Emotion] Monitor started (every {:.0f}s)".format(self._interval))

    def stop(self):
        self._running = False

    @property
    def state(self) -> tuple:
        return self._latest_label, self._latest_conf

    def _loop(self):
        sample_idx = 0
        while self._running:
            # Only run inference if we have REAL EEG data
            if self._eeg_data is None or len(self._eeg_data) == 0:
                time.sleep(self._interval)
                continue
            try:
                sample = self._eeg_data[sample_idx % len(self._eeg_data)]
                if sample.ndim == 2:
                    sample = sample[None, None, :, :]
                elif sample.ndim == 3:
                    sample = sample[None, :, :, :]
                sample = sample.astype(np.float32)

                outputs = self._rknn.inference(inputs=[sample])
                logits = outputs[1][0] if len(outputs) > 1 else outputs[0][0]
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                top = int(probs.argmax())

                self._latest_label = self.EMOTIONS[top] if top < 3 else "neutral"
                self._latest_conf = float(probs[top])
                ctx.set_emotion(self._latest_label, self._latest_conf)
                sample_idx += 1
            except Exception as e:
                print("[Emotion] Inference error: {}".format(e))
            time.sleep(self._interval)

    def is_loaded(self) -> bool:
        return self._rknn is not None

    def cleanup(self):
        if self._rknn:
            self._rknn.release()

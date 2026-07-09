"""Emotion Skill — EEG-Conformer emotion recognition on NPU."""
import time, os, numpy as np
from .base import Skill


class EmotionSkill(Skill):
    name = "emotion"
    description = "EEG emotion recognition - classifies brain signals into 3 emotions using NPU"
    requires_camera = False
    requires_eeg = True

    def __init__(self, chat_skill=None):
        self._rknn = None
        self._chat = chat_skill

    def load(self) -> bool:
        from rknnlite.api import RKNNLite
        model = "/home/elf/Projects/emotions/EEG-Conformer/eeg_conformer.rknn"
        if not os.path.isfile(model):
            print("[Emotion] Model not found:", model)
            return False
        self._rknn = RKNNLite()
        if self._rknn.load_rknn(model) != 0:
            print("[Emotion] load_rknn failed")
            return False
        if self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            print("[Emotion] init_runtime failed")
            return False
        print("[Emotion] EEG-Conformer loaded on NPU")
        return True

    def is_ready(self) -> bool:
        return self._rknn is not None

    def execute(self, question: str, **kwargs) -> str:
        eeg_path = "/home/elf/Projects/emotions/S1_session1.npy"
        if not os.path.isfile(eeg_path):
            return "[Emotion] No EEG data file found."

        try:
            data = np.load(eeg_path, allow_pickle=True)
        except Exception as e:
            return "[Emotion] Failed to load EEG data: {}".format(e)

        # Take first sample
        sample = data[0]
        if len(sample.shape) == 2:
            sample = sample[None, None, :, :]  # (8,200) -> (1,1,8,200)
        elif len(sample.shape) == 3:
            sample = sample[None, :, :, :]     # (1,8,200) -> (1,1,8,200)
        sample = sample.astype(np.float32)

        outputs = self._rknn.inference(inputs=[sample])
        logits = outputs[1][0] if len(outputs) > 1 else outputs[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        emotions = ["Neutral", "Positive", "Negative"]
        top_idx = int(probs.argmax())
        top_conf = float(probs[top_idx])
        label = emotions[top_idx] if top_idx < len(emotions) else "Unknown"

        result = "EEG analysis: {} ({:.0%}). ".format(label, top_conf)

        if self._chat:
            return self._chat.execute(result + question)
        return result + question

    def cleanup(self):
        if self._rknn:
            self._rknn.release()

"""
Shared Context — global state updated by EEG monitors, read by LLM.
"""
import threading


class Context:
    """Thread-safe singleton holding user state from EEG monitors."""

    def __init__(self):
        self._lock = threading.Lock()
        self._attention = 0.5    # 0.0=focused, 1.0=relaxed
        self._emotion = "neutral"  # neutral/positive/negative
        self._emotion_conf = 0.0
        self._updated = False

    # ---- Attention ----
    @property
    def attention(self) -> float:
        with self._lock:
            return self._attention

    def set_attention(self, value: float):
        with self._lock:
            self._attention = max(0.0, min(1.0, value))
            self._updated = True

    # ---- Emotion ----
    @property
    def emotion(self) -> str:
        with self._lock:
            return self._emotion

    @property
    def emotion_conf(self) -> float:
        with self._lock:
            return self._emotion_conf

    def set_emotion(self, label: str, confidence: float):
        with self._lock:
            self._emotion = label
            self._emotion_conf = confidence
            self._updated = True

    # ---- System prompt generation ----
    def get_system_prompt(self) -> str:
        """Build a context-aware system prompt for the LLM."""
        with self._lock:
            att = self._attention
            emo = self._emotion

        parts = ["You are an AI assistant on an embedded device (ELF2 RK3588)."]

        # Attention-based guidance
        if att < 0.3:
            parts.append("The driver is HIGHLY FOCUSED. Keep answers extremely short and direct. No small talk.")
        elif att > 0.7:
            parts.append("The driver is relaxed. You can be more conversational and detailed.")
        else:
            parts.append("The driver is in a normal state. Balance brevity with completeness.")

        # Emotion-based guidance
        if emo == "negative":
            parts.append("The driver seems stressed. Use calming, reassuring language. Prioritize safety advice.")
        elif emo == "positive":
            parts.append("The driver is in a good mood. Be friendly and upbeat.")

        return " ".join(parts)


# Global singleton
ctx = Context()

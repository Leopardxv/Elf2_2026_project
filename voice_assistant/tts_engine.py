#!/usr/bin/env python3
"""
TTS Engine — Text-to-speech using sherpa-onnx OfflineTts with VITS-Melo zh-en model.

Pre-generates "我在" response audio at load time for instant wake confirmation.
"""
import os
import glob
import numpy as np
import sherpa_onnx

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "sherpa", "tts")


def _find_model_files(model_dir: str):
    def _find(pattern):
        files = sorted(glob.glob(os.path.join(model_dir, pattern)))
        if not files:
            raise FileNotFoundError(f"No file matching '{pattern}' in {model_dir}")
        return files[0]

    model = _find("*model*.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")
    if not os.path.isfile(tokens):
        # try speakers.txt for icefall-style models
        alt = os.path.join(model_dir, "speakers.txt")
        if os.path.isfile(alt):
            tokens = alt
    lexicon = os.path.join(model_dir, "lexicon.txt")
    if not os.path.isfile(lexicon):
        lexicon = ""
    dict_dir = os.path.join(model_dir, "dict")
    if not os.path.isdir(dict_dir):
        dict_dir = ""

    return model, tokens, lexicon, dict_dir


class TTSEngine:
    """
    Chinese text-to-speech via VITS-Melo.

    Usage:
        tts = TTSEngine()
        tts.load()
        audio = tts.synthesize("你好世界")
        tts.play(audio)
    """

    def __init__(self, model_dir: str = None):
        self._model_dir = model_dir or MODEL_DIR
        self._tts: sherpa_onnx.OfflineTts = None
        self._sample_rate = 22050
        self._wozai_audio: np.ndarray = None

    def load(self) -> bool:
        model, tokens, lexicon, dict_dir = _find_model_files(self._model_dir)

        mc = sherpa_onnx.OfflineTtsModelConfig()
        mc.num_threads = 2
        mc.vits.model = model
        mc.vits.tokens = tokens
        if lexicon:
            mc.vits.lexicon = lexicon
        if dict_dir:
            mc.vits.dict_dir = dict_dir
        mc.vits.noise_scale = 0.667
        mc.vits.noise_scale_w = 0.8
        mc.vits.length_scale = 1.0

        tts_config = sherpa_onnx.OfflineTtsConfig(mc, max_num_sentences=1)
        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self._sample_rate = self._tts.sample_rate

        self._wozai_audio = self.synthesize("我在")
        print(f"[TTS] Loaded (sr={self._sample_rate})")
        return True

    def synthesize(self, text: str, sid: int = 0, speed: float = 1.0) -> np.ndarray:
        """
        Convert text to speech audio.

        Args:
            text: Chinese/English text to speak.
            sid: Speaker ID (0 = default).
            speed: Playback speed multiplier.

        Returns:
            float32 [-1,1] mono numpy array.
        """
        if not self._tts:
            raise RuntimeError("[TTS] Model not loaded.")
        if not text.strip():
            return np.array([], dtype=np.float32)

        result = self._tts.generate(text, sid=sid, speed=speed)
        if result is None or result.samples is None:
            return np.array([], dtype=np.float32)
        return np.array(result.samples, dtype=np.float32)

    @property
    def wozai_audio(self) -> np.ndarray:
        return self._wozai_audio

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def loaded(self) -> bool:
        return self._tts is not None

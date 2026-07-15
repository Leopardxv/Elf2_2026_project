#!/usr/bin/env python3
"""
TTS Engine — Text-to-speech using sherpa-onnx OfflineTts with VITS-Melo zh-en model.

Pre-generates "我在" response audio at load time for instant wake confirmation.
"""
import os
import glob
import hashlib
import numpy as np
import sherpa_onnx

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "sherpa", "tts")
CACHE_MAX_FILES = 64
CACHE_MAX_BYTES = 128 * 1024 * 1024
TTS_CACHE_ENABLED = os.getenv("VOICE_TTS_CACHE", "0").strip().lower() in {"1", "true", "yes"}


def _find_model_files(model_dir: str):
    def _find(pattern):
        files = sorted(glob.glob(os.path.join(model_dir, pattern)))
        if not files:
            raise FileNotFoundError(f"No file matching '{pattern}' in {model_dir}")
        return files[0]

    exact_model = os.path.join(model_dir, "model.onnx")
    model = exact_model if os.path.isfile(exact_model) else _find("*model*.onnx")
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
        self._model_path = ""
        self._cache_dir = os.path.join(self._model_dir, "cache")
        self._cache_enabled = TTS_CACHE_ENABLED

    def load(self) -> bool:
        model, tokens, lexicon, dict_dir = _find_model_files(self._model_dir)
        self._model_path = model
        if self._cache_enabled:
            os.makedirs(self._cache_dir, exist_ok=True)

        mc = sherpa_onnx.OfflineTtsModelConfig()
        # TTS runs on CPU while the LLM stays on NPU.  Six threads reduce
        # offline VITS synthesis time without adding a resident model.
        mc.num_threads = 6
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

    def synthesize(self, text: str, sid: int = 0, speed: float = 0.9) -> np.ndarray:
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

        cache_path = ""
        if self._cache_enabled:
            cache_path = self._cache_path(text, sid, speed)
            cached_audio = self.get_cached_audio(text, sid, speed)
            if cached_audio is not None:
                return cached_audio

        result = self._tts.generate(text, sid=sid, speed=speed)
        if result is None or result.samples is None:
            return np.array([], dtype=np.float32)
        audio = np.array(result.samples, dtype=np.float32)
        if self._cache_enabled:
            self._store_cache(cache_path, audio)
        return audio

    def get_cached_audio(self, text: str, sid: int = 0,
                         speed: float = 1.0) -> np.ndarray | None:
        """Return an exact cached waveform without invoking the TTS model."""
        if not self._cache_enabled:
            return None
        cache_path = self._cache_path(text, sid, speed)
        if not cache_path or not os.path.isfile(cache_path):
            return None
        try:
            audio = np.load(cache_path, allow_pickle=False)
            os.utime(cache_path, None)
            print(f"[TTS] Cache hit ({len(text)} chars)")
            return np.asarray(audio, dtype=np.float32)
        except (OSError, ValueError):
            try:
                os.unlink(cache_path)
            except OSError:
                pass
            return None

    def _cache_path(self, text: str, sid: int, speed: float) -> str:
        if not self._cache_enabled or not self._model_path or len(text) > 100:
            return ""
        try:
            stat = os.stat(self._model_path)
        except OSError:
            return ""
        fingerprint = f"{os.path.basename(self._model_path)}:{stat.st_size}:{stat.st_mtime_ns}"
        payload = f"{fingerprint}:{self._sample_rate}:{sid}:{speed:.4f}:{text}"
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return os.path.join(self._cache_dir, f"{key}.npy")

    def _store_cache(self, cache_path: str, audio: np.ndarray):
        if not cache_path or audio.size == 0:
            return
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        try:
            with open(tmp_path, "wb") as f:
                np.save(f, audio, allow_pickle=False)
            os.replace(tmp_path, cache_path)
            self._prune_cache()
        except OSError:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _prune_cache(self):
        try:
            files = [
                (os.path.getmtime(path), os.path.getsize(path), path)
                for path in glob.glob(os.path.join(self._cache_dir, "*.npy"))
            ]
        except OSError:
            return
        total_bytes = sum(size for _, size, _ in files)
        files.sort()
        while files and (len(files) > CACHE_MAX_FILES or total_bytes > CACHE_MAX_BYTES):
            _, size, path = files.pop(0)
            try:
                os.unlink(path)
                total_bytes -= size
            except OSError:
                pass

    @property
    def wozai_audio(self) -> np.ndarray:
        return self._wozai_audio

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def loaded(self) -> bool:
        return self._tts is not None

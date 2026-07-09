#!/usr/bin/env python3
"""
情绪安抚音乐生成器 — 检测到负面情绪时播放舒缓音乐。
"""
import numpy as np
import wave
import subprocess
import tempfile
import os
import threading


def _play(filepath: str, timeout: int = 20):
    """非阻塞播放 WAV 文件"""
    def _run():
        subprocess.run(
            ["aplay", "-q", "-D", "pulse",
             "--buffer-time=80000", "--period-time=16000", filepath],
            timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def generate_relaxing_music(duration_sec: float = 15, sample_rate: int = 44100) -> str:
    """
    生成舒缓钢琴风格环境音乐，返回临时 WAV 文件路径。
    调用方负责播完后 os.unlink。
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    music = np.zeros_like(t, dtype=np.float64)

    # 温和的和弦进行：C → Am → F → G
    chords = [
        (261.6, 329.6, 392.0),   # C major
        (220.0, 261.6, 329.6),   # Am
        (174.6, 220.0, 261.6),   # F major
        (196.0, 246.9, 293.7),   # G major
    ]

    for ci, chord in enumerate(chords):
        start = ci * (duration_sec / len(chords))
        end = min(start + duration_sec / len(chords) + 0.5, duration_sec)
        idx = (t >= start) & (t < end)
        env = np.clip(np.sin(np.pi * (t[idx] - start) / (end - start)), 0, 1)
        for freq in chord:
            music[idx] += 0.10 * env * np.sin(2 * np.pi * freq * t[idx])

    # 高音装饰
    high_notes = [523.3, 587.3, 440.0, 493.9]
    for ci, freq in enumerate(high_notes):
        start = ci * (duration_sec / len(high_notes)) + 1.0
        end = min(start + 2.0, duration_sec)
        idx = (t >= start) & (t < end)
        env = np.clip(np.sin(np.pi * (t[idx] - start) / 2.0), 0, 1)
        music[idx] += 0.05 * env * np.sin(2 * np.pi * freq * t[idx])

    music = music / np.abs(music).max() * 0.8
    music_f32 = music.astype(np.float32)

    # Write WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((music_f32 * 32767).astype(np.int16).tobytes())

    return tmp


def play_music(duration: float = 15) -> threading.Thread:
    """生成并播放舒缓音乐，返回播放线程。"""
    tmp = generate_relaxing_music(duration)
    t = _play(tmp, timeout=int(duration) + 5)
    # 延迟删除文件（等播放完）
    def _cleanup():
        t.join()
        try:
            os.unlink(tmp)
        except OSError:
            pass
    threading.Thread(target=_cleanup, daemon=True).start()
    return t


# ================================================================
# 情绪追踪
# ================================================================
_emotion_history: list = []  # 最近的情绪标签
_MAX_HISTORY = 5
_NEGATIVE_THRESHOLD = 3  # 连续 N 次消极触发安抚


def record_emotion(label: str):
    """记录一次情绪检测结果"""
    global _emotion_history
    _emotion_history.append(label)
    if len(_emotion_history) > _MAX_HISTORY:
        _emotion_history.pop(0)


def should_console() -> bool:
    """检查是否应该触发情绪安抚"""
    if len(_emotion_history) < _NEGATIVE_THRESHOLD:
        return False
    recent = _emotion_history[-_NEGATIVE_THRESHOLD:]
    return all("消极" in e for e in recent)

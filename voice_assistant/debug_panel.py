#!/usr/bin/env python3
"""
ELF2 语音助手诊断面板 — PyQt5 实时调试工具

显示: 麦克风电平 | KWS唤醒状态 | ASR识别文字 | LLM回复 | TTS输出
双击桌面图标启动
"""
import sys
import os
import time
import threading
import queue
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit, QGroupBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette

import sounddevice as sd
import vosk

# —— Model paths ——
MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
VOSK_MODEL = os.path.join(MODELS, "vosk", "vosk-model-small-cn-0.22")
LLM_MODEL = os.path.join(MODELS, "qwen2.5-0.5b-instruct-q2_k.gguf")
SAMPLE_RATE = 16000
MIC_DEVICE = os.getenv("VOICE_MIC_DEVICE", "default")
MIC_CHANNEL = 0

WAKE_KEYWORD = "精灵"
ENERGY_THRESHOLD = 0.03
SILENCE_SEC = 1.2
MIN_SPEECH_SEC = 0.4
MAX_UTTERANCE_SEC = 10.0
VAD_CHUNK_SEC = 0.15


class LogSignals(QObject):
    log = pyqtSignal(str)
    audio_level = pyqtSignal(float)
    status = pyqtSignal(str, str)  # stage, color
    asr_text = pyqtSignal(str)
    llm_text = pyqtSignal(str)
    tts_text = pyqtSignal(str)
    kws_fired = pyqtSignal()


class VoiceDebugPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("精灵精灵 — 语音助手诊断面板")
        self.resize(900, 700)

        self.signals = LogSignals()
        self.signals.log.connect(self._append_log)
        self.signals.audio_level.connect(self._update_level)
        self.signals.status.connect(self._set_status)
        self.signals.asr_text.connect(self._set_asr)
        self.signals.llm_text.connect(self._set_llm)
        self.signals.tts_text.connect(self._set_tts)
        self.signals.kws_fired.connect(self._on_kws)

        self._running = False
        self._kws = None
        self._asr = None
        self._tts = None
        self._llm = None
        self._mic_stream = None
        self._kws_stream = None

        self._build_ui()
        self._log("诊断面板启动。点击 [▶ 启动] 加载模型并开始监听。")

    # ================================================================
    #  UI
    # ================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        # -- Row 0: Status bar --
        self.status_group = QGroupBox("状态")
        sl = QHBoxLayout(self.status_group)
        self.status_labels = {}
        for name, label_text in [
            ("idle", "⏳ 待机"), ("kws", "👂 听唤醒词"), ("woke", "🔔 已唤醒"),
            ("recording", "🎙️ 录音中"), ("thinking", "🧠 思考中"), ("speaking", "🔊 播放中")
        ]:
            lbl = QLabel(label_text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFont(QFont("Sans", 13, QFont.Bold))
            lbl.setStyleSheet("padding:6px;border-radius:6px;background:#333;color:#888;")
            sl.addWidget(lbl)
            self.status_labels[name] = lbl
        layout.addWidget(self.status_group)

        # -- Row 1: Audio level --
        level_group = QGroupBox("麦克风电平")
        ll = QHBoxLayout(level_group)
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(28)
        self.level_bar.setStyleSheet(
            "QProgressBar{border:1px solid #555;border-radius:4px;background:#222}"
            "QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0f0,stop:0.7 #ff0,stop:1 #f00);border-radius:3px}"
        )
        ll.addWidget(self.level_bar)
        self.level_pct = QLabel("0%")
        self.level_pct.setFixedWidth(50)
        ll.addWidget(self.level_pct)
        layout.addWidget(level_group)

        # -- Row 2: Control buttons --
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ 启动")
        self.btn_start.clicked.connect(self._start)
        self.btn_start.setFixedHeight(40)
        self.btn_start.setStyleSheet("QPushButton{background:#2a2;color:#fff;font-size:16px;font-weight:bold;border-radius:6px;padding:0 30px}")
        btn_layout.addWidget(self.btn_start)

        self.btn_speak = QPushButton("🔊 播'我在'")
        self.btn_speak.clicked.connect(self._test_speaker)
        self.btn_speak.setFixedHeight(40)
        self.btn_speak.setStyleSheet("QPushButton{background:#36a;color:#fff;font-size:14px;border-radius:6px;padding:0 20px}")
        btn_layout.addWidget(self.btn_speak)

        self.btn_mic = QPushButton("🎤 测麦克风")
        self.btn_mic.clicked.connect(self._test_mic)
        self.btn_mic.setFixedHeight(40)
        self.btn_mic.setStyleSheet("QPushButton{background:#555;color:#fff;font-size:14px;border-radius:6px;padding:0 20px}")
        btn_layout.addWidget(self.btn_mic)

        layout.addLayout(btn_layout)

        # -- Row 3: Text displays --
        text_layout = QHBoxLayout()

        # ASR
        asr_group = QGroupBox("ASR 识别结果 (你说的话)")
        asr_l = QVBoxLayout(asr_group)
        self.asr_display = QLabel("等待启动...")
        self.asr_display.setWordWrap(True)
        self.asr_display.setFont(QFont("Sans", 16, QFont.Bold))
        self.asr_display.setStyleSheet("color:#0f0;background:#111;padding:10px;border-radius:6px;min-height:60px")
        self.asr_display.setAlignment(Qt.AlignCenter)
        asr_l.addWidget(self.asr_display)
        text_layout.addWidget(asr_group)

        # LLM
        llm_group = QGroupBox("LLM 回复 (AI回答)")
        llm_l = QVBoxLayout(llm_group)
        self.llm_display = QLabel("等待启动...")
        self.llm_display.setWordWrap(True)
        self.llm_display.setFont(QFont("Sans", 14))
        self.llm_display.setStyleSheet("color:#ff0;background:#111;padding:10px;border-radius:6px;min-height:60px")
        self.llm_display.setAlignment(Qt.AlignCenter)
        llm_l.addWidget(self.llm_display)
        text_layout.addWidget(llm_group)

        layout.addLayout(text_layout)

        # TTS
        tts_group = QGroupBox("TTS 输出文字 (扬声器播放)")
        tts_l = QVBoxLayout(tts_group)
        self.tts_display = QLabel("等待启动...")
        self.tts_display.setWordWrap(True)
        self.tts_display.setFont(QFont("Sans", 13))
        self.tts_display.setStyleSheet("color:#f8f;background:#111;padding:8px;border-radius:6px;min-height:30px")
        self.tts_display.setAlignment(Qt.AlignCenter)
        tts_l.addWidget(self.tts_display)
        layout.addWidget(tts_group)

        # -- Row 4: Log --
        log_group = QGroupBox("日志")
        log_l = QVBoxLayout(log_group)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Monospace", 9))
        self.log_display.setStyleSheet("background:#111;color:#aaa;padding:5px;border-radius:4px")
        self.log_display.setFixedHeight(150)
        log_l.addWidget(self.log_display)
        layout.addWidget(log_group)

        # Status bar
        self.statusBar().showMessage("就绪")

    # ================================================================
    #  Slots
    # ================================================================
    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{ts}] {msg}")

    def _append_log(self, msg):
        self._log(msg)

    def _update_level(self, val):
        pct = min(int(val * 100 / 0.3), 100) if val < 0.3 else 100
        self.level_bar.setValue(pct)
        self.level_pct.setText(f"{pct}%")

    def _set_status(self, stage, color):
        for name, lbl in self.status_labels.items():
            if name == stage:
                lbl.setStyleSheet(f"padding:6px;border-radius:6px;background:{color};color:#fff;")
            else:
                lbl.setStyleSheet("padding:6px;border-radius:6px;background:#333;color:#888;")

    def _set_asr(self, text):
        self.asr_display.setText(text or "(无声)")

    def _set_llm(self, text):
        self.llm_display.setText(text or "等待中...")

    def _set_tts(self, text):
        self.tts_display.setText(text or "")

    def _on_kws(self):
        self.signals.status.emit("woke", "#e74c3c")
        self._log("🔔 唤醒词检测到！正在处理...")

    # ================================================================
    #  Start / Stop
    # ================================================================
    def _start(self):
        if self._running:
            return
        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_start.setText("加载中...")
        self._log("加载模型...")

        def _load():
            try:
                # TTS
                self._log("加载 TTS...")
                from voice_assistant.tts_engine import TTSEngine
                self._tts = TTSEngine()
                self._tts.load()
                self._log(f"  TTS loaded ({self._tts.sample_rate}Hz)")

                # ASR
                self._log("加载 ASR...")
                from voice_assistant.asr_engine import ASREngine
                self._asr = ASREngine()
                self._asr.load()
                self._log("  ASR loaded")

                # KWS
                self._log("加载 KWS (vosk)...")
                vosk.SetLogLevel(-1)
                self._kws_model = vosk.Model(VOSK_MODEL)
                self._log("  KWS loaded")

                # Mic
                self._start_mic_monitor()

                self.signals.status.emit("kws", "#3498db")
                self.btn_start.setText("✅ 运行中")
                self.btn_start.setStyleSheet("QPushButton{background:#28a;color:#fff;font-size:16px;font-weight:bold;border-radius:6px;padding:0 30px}")
                self._log("✅ 全部就绪！说 '精灵精灵' 唤醒")
            except Exception as e:
                self._log(f"❌ 错误: {e}")
                import traceback; traceback.print_exc()
                self._running = False
                self.btn_start.setEnabled(True)
                self.btn_start.setText("▶ 启动")

        threading.Thread(target=_load, daemon=True).start()

    def _stop(self):
        self._running = False
        if self._mic_stream:
            self._mic_stream.stop()
            self._mic_stream.close()
            self._mic_stream = None
        if self._llm:
            self._llm.unload()
            self._llm = None
        self.signals.status.emit("idle", "#555")
        self._log("已停止")

    def closeEvent(self, event):
        self._stop()
        event.accept()

    # ================================================================
    #  Mic monitor + vosk KWS
    # ================================================================
    def _start_mic_monitor(self):
        chunk_frames = int(VAD_CHUNK_SEC * SAMPLE_RATE)
        silence_thresh = int(SILENCE_SEC / VAD_CHUNK_SEC)
        min_speech_frames = int(MIN_SPEECH_SEC / VAD_CHUNK_SEC)
        max_chunks = int(MAX_UTTERANCE_SEC / VAD_CHUNK_SEC)

        kws_model = self._kws_model
        buffer = []
        speech_count = 0
        silence_count = 0
        in_speech = False

        def _transcribe_kws(audio_i16):
            rec = vosk.KaldiRecognizer(kws_model, SAMPLE_RATE)
            for i in range(0, len(audio_i16), 4000):
                chunk = audio_i16[i:i+4000].tobytes()
                rec.AcceptWaveform(chunk)
            result = json.loads(rec.FinalResult())
            return result.get("text", "")

        def audio_cb(indata, frames, t, status):
            nonlocal in_speech, speech_count, silence_count, buffer
            if not self._running:
                return
            ch = indata[:, MIC_CHANNEL] if indata.shape[1] > 1 else indata[:, 0]
            chunk_f32 = ch.ravel().astype(np.float32)
            chunk_i16 = (np.clip(chunk_f32, -1.0, 1.0) * 32767).astype(np.int16)
            energy = float(np.sqrt(np.mean(chunk_f32 ** 2)))
            self.signals.audio_level.emit(energy)

            if energy > ENERGY_THRESHOLD:
                if not in_speech:
                    in_speech = True
                    speech_count = 0
                    silence_count = 0
                    keep = min(len(buffer), int(0.5 / VAD_CHUNK_SEC))
                    buffer = buffer[-keep:] if keep else []
                buffer.append(chunk_i16)
                speech_count += 1
                silence_count = 0
            elif in_speech:
                buffer.append(chunk_i16)
                silence_count += 1
                ended = (
                    speech_count >= min_speech_frames and
                    (silence_count >= silence_thresh or len(buffer) >= max_chunks)
                )
                if ended:
                    audio = np.concatenate(buffer)
                    buffer.clear()
                    in_speech = False
                    if speech_count >= min_speech_frames:
                        text = _transcribe_kws(audio)
                        if text:
                            self._log(f"[KWS] Heard: '{text}'")
                            if WAKE_KEYWORD in text:
                                self._log("🔔 唤醒词!")
                                self.signals.kws_fired.emit()
                                self._handle_wake()

        self._mic_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_frames,
            device=MIC_DEVICE,
            callback=audio_cb,
        )
        self._mic_stream.start()

    def _interaction_flow(self):
        try:
            # Stop mic for ASR
            if self._mic_stream:
                self._mic_stream.stop()
            time.sleep(0.3)

            # Play "我在"
            self._log("🔊 播放 '我在'...")
            self.signals.status.emit("speaking", "#e67e22")
            self.signals.tts_text.emit("我在")
            from voice_assistant.audio_io import AudioPlayer
            ap = AudioPlayer()
            ap.play(self._tts.wozai_audio, self._tts.sample_rate)
            time.sleep(0.2)

            # ASR
            self.signals.status.emit("recording", "#2ecc71")
            self._log("🎙️ 录音中，请说话...")
            text = self._asr.listen_and_transcribe(timeout=12.0)
            self.signals.asr_text.emit(text or "(未识别)")
            self._log(f"ASR 结果: '{text}'")

            if not text:
                self._speak_fallback("我没有听清")
                return

            # LLM
            self.signals.status.emit("thinking", "#9b59b6")
            self._log("🧠 LLM 推理中...")
            answer = self._run_llm(text)
            self.signals.llm_text.emit(answer or "(无回复)")
            self._log(f"LLM: '{answer}'" if answer else "LLM: 无回复")

            if not answer:
                self._speak_fallback("我想不出答案")
                return

            # TTS + Play
            self.signals.status.emit("speaking", "#e67e22")
            self.signals.tts_text.emit(answer)
            self._log(f"🔊 播放中...")
            audio = self._tts.synthesize(answer)
            ap.play(audio, self._tts.sample_rate)

        except Exception as e:
            self._log(f"❌ 交互错误: {e}")
        finally:
            # Resume KWS
            time.sleep(0.3)
            if self._mic_stream:
                self._mic_stream.start()
            self.signals.status.emit("kws", "#3498db")
            self._log("👂 继续监听唤醒词")

    def _speak_fallback(self, text):
        self.signals.tts_text.emit(text)
        self.signals.status.emit("speaking", "#e67e22")
        from voice_assistant.audio_io import AudioPlayer
        try:
            audio = self._tts.synthesize(text)
            AudioPlayer().play(audio, self._tts.sample_rate)
        except Exception:
            pass

    def _run_llm(self, question):
        if not self._llm:
            try:
                from dispatch.llama_engine import LlamaEngine
                self._llm = LlamaEngine()
                self._llm.load(LLM_MODEL)
                self._log("  LLM 已加载")
            except Exception as e:
                self._log(f"  LLM 加载失败: {e}")
                return None
        if self._llm and self._llm.is_loaded():
            return self._llm.chat(question)
        return None

    # ================================================================
    #  Test buttons
    # ================================================================
    def _test_speaker(self):
        self._log("🔊 测试扬声器 (DP)...")
        try:
            import numpy as np
            from voice_assistant.audio_io import AudioPlayer
            sr = 44100
            t = np.linspace(0, 0.5, int(0.5 * sr), False)
            tone = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            AudioPlayer().play(tone, sample_rate=sr)
            self._log("  扬声器 OK (440Hz 提示音)")
        except Exception as e:
            self._log(f"  扬声器失败: {e}")

    def _test_mic(self):
        self._log("🎤 测试麦克风 (3秒)...")
        try:
            audio = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16', device=MIC_DEVICE)
            sd.wait()
            ch1 = audio[:, MIC_CHANNEL] if audio.shape[1] > 1 else audio[:, 0]
            peak = float(np.abs(ch1).max())
            self._log(f"  麦克风 OK, 峰值={peak}")
            self.level_bar.setValue(min(int(peak * 100 / 32767 * 3), 100))
        except Exception as e:
            self._log(f"  麦克风失败: {e}")


def main():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    dark = QPalette()
    dark.setColor(QPalette.Window, QColor(30, 30, 30))
    dark.setColor(QPalette.WindowText, QColor(200, 200, 200))
    dark.setColor(QPalette.Base, QColor(25, 25, 25))
    dark.setColor(QPalette.Text, QColor(200, 200, 200))
    app.setPalette(dark)

    win = VoiceDebugPanel()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

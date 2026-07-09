#!/usr/bin/env python3
"""
Voice Agent — Main orchestrator for the "精灵精灵" voice assistant.

Pipeline:
  KWS ("精灵精灵") → Play "我在" → ASR (record command) → LLM → TTS → Play response

Usage:
    conda activate eeg
    cd ~/Projects
    PYTHONPATH="$HOME/Projects:$PYTHONPATH" python3 -m voice_assistant.voice_agent
"""
import os
import sys
import time
import threading
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from voice_assistant.kws_engine import WakeWordEngine
from voice_assistant.asr_engine import ASREngine
from voice_assistant.tts_engine import TTSEngine
from voice_assistant.knowledge import KnowledgeBase
from voice_assistant import music


class VoiceAgent:
    """
    Voice-controlled AI assistant.

    States: WAITING → WOKE → RECORDING → THINKING → SPEAKING → WAITING
    """

    def __init__(self):
        self.kws = WakeWordEngine()
        self.asr = ASREngine()
        self.tts = TTSEngine()
        self.llm = None
        self.kb = KnowledgeBase()

        self._wake_lock = threading.Lock()
        self._busy = threading.Event()
        self._neg_count = 0  # 连续消极情绪计数

    # ----------------------------------------------------------------
    #  Lifecycle
    # ----------------------------------------------------------------
    def start(self):
        """Load all models and begin listening for wake word."""
        print("=" * 55)
        print("精灵精灵 — Voice Agent Boot")
        print("=" * 55)

        # Start ROS2 sensor receiver in background (non-blocking)
        try:
            import threading as _th
            _th.Thread(target=self._start_sensor_receiver, daemon=True).start()
        except Exception:
            print("[Agent] Sensor receiver will be skipped")

        print("\n[1/4] TTS (VITS-Melo)...")
        self.tts.load()
        print("[2/4] ASR (vosk CN)...")
        self.asr.load()
        print("[3/4] KWS (vosk VAD)...")
        self.kws.load()
        print("[4/4] LLM (ChatSkill NPU)...")
        self._load_llm()

        print("\n" + "=" * 55)
        print("Agent ready. Say '精灵精灵' to wake me up!")
        print("=" * 55 + "\n")

        self.kws.start(on_wake=self._on_wake)

    @staticmethod
    def _start_sensor_receiver():
        """Start ROS2 sensor receiver (runs in background thread)."""
        try:
            import rclpy
            from xiaoche_communication.sensor_receiver import SensorReceiver, get_sensor_state
            rclpy.init(args=[])
            node = SensorReceiver(state=get_sensor_state())
            print("[Sensor] Listening for gas/pointcloud from A733...")
            rclpy.spin(node)
        except Exception as e:
            print(f"[Sensor] Not available: {e}")

    def stop(self):
        """Stop all engines and release resources."""
        print("[Agent] Shutting down...")
        self._busy.set()
        self.kws.stop()
        if self.llm:
            self.llm.unload()
        print("[Agent] Goodbye.")

    # ----------------------------------------------------------------
    #  Wake callback (called from KWS audio thread — must be fast)
    # ----------------------------------------------------------------
    def _on_wake(self):
        """Handle wake word detection."""
        with self._wake_lock:
            if self._busy.is_set():
                return
            self._busy.set()
        threading.Thread(target=self._handle_interaction, daemon=True).start()

    # ----------------------------------------------------------------
    #  Interaction flow
    # ----------------------------------------------------------------
    def _handle_interaction(self):
        """Full interaction: wake → play "我在" → ASR → LLM → TTS → restart KWS."""
        # Stop KWS to free mic
        self.kws.stop()
        time.sleep(0.5)  # Let PortAudio fully release mic

        try:
            # 1. Play pre-generated "我在"
            print("[Agent] Wake! Playing 'wo zai'...")
            self._play_wozai()
            time.sleep(0.2)

            # 2. ASR recording
            question = self.asr.listen_and_transcribe(timeout=8.0)

            if not question:
                self._speak("我没有听清，请再说一遍")
                return

            print(f"[Agent] User: {question}")

            # 情绪追踪
            neg_keywords = ['难过', '不开心', '低落', '沮丧', '烦躁', '焦虑', '压抑', '想哭']
            if any(k in question for k in neg_keywords):
                self._neg_count += 1
                print(f"[Agent] Negative emotion: {self._neg_count}/3")
            else:
                self._neg_count = max(0, self._neg_count - 1)

            # LLM inference (with knowledge base)
            answer = self._ask_llm(question)
            if not answer:
                self._speak("我想不出答案")
                return

            print(f"[Agent] LLM: {answer}")

            # 5. TTS + Play
            self._speak(answer)

            # 情绪安抚：连续 3 次消极触发音乐
            if self._neg_count >= 3:
                self._neg_count = 0
                time.sleep(0.3)
                self._speak("检测到您长时间情绪低落，为您播放舒缓音乐。")
                music.play_music(duration=15)

        except Exception as e:
            print(f"[Agent] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restart KWS
            time.sleep(0.2)
            self._busy.clear()
            self.kws.reset()
            self.kws.start(on_wake=self._on_wake)

    # ----------------------------------------------------------------
    #  Sub-components
    # ----------------------------------------------------------------
    def _load_llm(self):
        try:
            from dispatch.skills.chat import ChatSkill
            self.llm = ChatSkill(use_npu=True)
            self.llm.load()  # NPU LLM
        except Exception as e:
            print(f"[Agent] LLM not available: {e}")
            self.llm = None

    def _ask_llm(self, question: str) -> str:
        if not self.llm or not self.llm.is_ready():
            return "语言模型未就绪"

        # 检索相关知识库内容
        matches = self.kb.search(question, top_k=2)

        # 附加传感器数据
        sensor_info = ""
        try:
            from xiaoche_communication.sensor_receiver import get_sensor_state
            state = get_sensor_state()
            gas = state.get_gas_summary()
            obs = state.get_obstacle_summary()
            if gas != "无气体数据" or obs != "未检测到障碍物":
                sensor_info = f"【当前环境传感器数据】\n气体: {gas}\n障碍物: {obs}\n"
        except Exception:
            pass

        if matches or sensor_info:
            know_parts = []
            if matches:
                know_parts.append("\n\n".join(f"【{t}】\n{c}" for t, c in matches))
            if sensor_info:
                know_parts.append(sensor_info)
            context = "\n\n".join(know_parts)
            prompt = (
                f"请参考以下信息回答问题，回答要专业、简洁、实用：\n\n"
                f"{context}\n\n"
                f"用户问题：{question}"
            )
            if matches:
                print(f"[Agent] KB match: {[t for t,_ in matches]}")
        else:
            prompt = question

        return self.llm.execute(prompt)

    def _speak(self, text: str):
        """Synthesize and play TTS audio."""
        print(f"[TTS] '{text}'")
        audio = self.tts.synthesize(text)
        if audio.size > 0:
            import subprocess, tempfile, wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            try:
                audio_i16 = (audio.astype(np.float32) * 32767).astype(np.int16)
                with wave.open(tmp, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2)
                    w.setframerate(self.tts.sample_rate)
                    w.writeframes(audio_i16.tobytes())
                subprocess.run(
                    ["aplay", "-q", "--buffer-time=100000", "--period-time=20000", tmp],
                    timeout=dur+5,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            finally:
                try: os.unlink(tmp)
                except OSError: pass

    def _play_wozai(self):
        """Play pre-generated '我在' audio."""
        audio = self.tts.wozai_audio
        if audio is not None and audio.size > 0:
            import subprocess, tempfile, wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            try:
                audio_i16 = (audio.astype(np.float32) * 32767).astype(np.int16)
                with wave.open(tmp, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2)
                    w.setframerate(self.tts.sample_rate)
                    w.writeframes(audio_i16.tobytes())
                subprocess.run(
                    ["aplay", "-q", "--buffer-time=100000", "--period-time=20000", tmp],
                    timeout=8,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            finally:
                try: os.unlink(tmp)
                except OSError: pass

    # ----------------------------------------------------------------
    #  Interactive mode
    # ----------------------------------------------------------------
    def run_forever(self):
        """Blocking main loop. Press Ctrl+C to stop."""
        self.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[Agent] Interrupted by user.")
        finally:
            self.stop()


def main():
    agent = VoiceAgent()
    agent.run_forever()


if __name__ == "__main__":
    main()

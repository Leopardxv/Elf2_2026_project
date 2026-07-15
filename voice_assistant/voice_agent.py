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
import re
import signal
import sys
import time
import threading
import codecs
import queue
import json
import subprocess
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
from bci.context_state import read_context

# None means stream every complete sentence.  The former 100-character cap
# silently dropped the tail of otherwise valid long answers.
MAX_VOICE_RESPONSE_CHARS = None


class SentenceSpeechStream:
    """Speak only complete model sentences while later tokens are generated."""

    def __init__(self, agent):
        self._agent = agent
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._text_queue = queue.Queue()
        self._audio_queue = queue.Queue()
        self._buffer = ""
        self._spoken_chars = 0
        self._queued = False
        self._synth_thread = threading.Thread(target=self._synthesize, daemon=True)
        self._play_thread = threading.Thread(target=self._play, daemon=True)
        self._synth_thread.start()
        self._play_thread.start()

    def on_chunk(self, chunk: bytes):
        self._append(self._decoder.decode(chunk))

    def finish(self) -> bool:
        self._append(self._decoder.decode(b"", final=True), flush=True)
        self._text_queue.put(None)
        return self._queued

    def wait(self):
        self._synth_thread.join()
        self._play_thread.join()

    def _append(self, text: str, flush: bool = False):
        self._buffer += text
        endings = "".join(chr(code) for code in (0x3002, 0xFF01, 0xFF1F, 0xFF1B))
        pauses = "".join(chr(code) for code in (0xFF0C, 0x3001, 0xFF1A))
        while self._buffer:
            sentence_positions = [self._buffer.find(mark) for mark in endings]
            sentence_positions = [position for position in sentence_positions if position >= 0]
            # A long clause ending at a comma is safe to synthesize early.
            # Include it even when a later full stop is already buffered.
            pause_positions = [self._buffer.find(mark) for mark in pauses]
            pause_positions = [position for position in pause_positions if position >= 11]
            positions = sentence_positions + pause_positions
            if not positions:
                break
            end = min(positions) + 1
            self._enqueue(self._buffer[:end])
            self._buffer = self._buffer[end:]
        if flush and self._buffer:
            self._enqueue(self._buffer)
            self._buffer = ""

    def _enqueue(self, text: str):
        text = self._agent._prepare_voice_text(text)
        if MAX_VOICE_RESPONSE_CHARS is not None:
            remaining = MAX_VOICE_RESPONSE_CHARS - self._spoken_chars
            if remaining <= 0:
                return
            if len(text) > remaining:
                text = self._agent._prepare_voice_text(text[:remaining])
        if text:
            self._spoken_chars += len(text)
            self._queued = True
            self._text_queue.put(text)

    def _synthesize(self):
        """Prepare later sentences while the playback thread is speaking."""
        try:
            while True:
                text = self._text_queue.get()
                if text is None:
                    return
                started = time.monotonic()
                audio = self._agent.tts.get_cached_audio(text)
                if audio is None:
                    audio = self._agent.tts.synthesize(text)
                print(f"[TTS] stream synth: {time.monotonic() - started:.2f}s")
                if audio is not None and audio.size:
                    self._audio_queue.put(audio)
        finally:
            self._audio_queue.put(None)

    def _play(self):
        """Keep one output stream open so sentence boundaries have no reset gap."""
        import subprocess

        process = None
        audio_started = None
        total_duration = 0.0
        try:
            while True:
                audio = self._audio_queue.get()
                if audio is None:
                    break
                if process is None:
                    process = subprocess.Popen(
                        [
                            "aplay", "-q", "-t", "raw", "-f", "S16_LE",
                            "-c", "1", "-r", str(self._agent.tts.sample_rate),
                            "--buffer-time=60000", "--period-time=15000",
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    preroll = self._agent._hfp_preroll_pcm()
                    if preroll.size:
                        process.stdin.write(preroll.tobytes())
                        total_duration += len(preroll) / self._agent.tts.sample_rate
                    audio_started = time.monotonic()

                pcm = self._agent._to_playback_pcm(audio)
                process.stdin.write(pcm.tobytes())
                process.stdin.flush()
                total_duration += len(pcm) / self._agent.tts.sample_rate

            if process:
                tail = self._agent._hfp_tail_pcm()
                if tail.size:
                    process.stdin.write(tail.tobytes())
                    total_duration += len(tail) / self._agent.tts.sample_rate
                process.stdin.close()
                elapsed = time.monotonic() - (audio_started or time.monotonic())
                process.wait(timeout=max(0.5, total_duration - elapsed + 0.75))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)
        except (BrokenPipeError, OSError, subprocess.SubprocessError):
            if process and process.poll() is None:
                process.kill()


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
        self._stop_event = threading.Event()
        self._shutting_down = threading.Event()
        self._interaction_thread = None
        self._music_lock = threading.Lock()
        self._music_requested = False
        self._last_emotion_at = 0.0
        self._negative_emotion_streak = 0
        self._last_music_at = 0.0

    # ----------------------------------------------------------------
    #  Lifecycle
    # ----------------------------------------------------------------
    def start(self):
        """Load all models and begin listening for wake word."""
        print("=" * 55)
        print("精灵精灵 — Voice Agent Boot")
        print("=" * 55)

        threading.Thread(target=self._start_sensor_receiver, daemon=True).start()
        threading.Thread(target=self._monitor_bci_emotion, daemon=True).start()

        print("\n[1/4] Vosk wake-word model...")
        import vosk
        vosk.SetLogLevel(-1)
        started = time.monotonic()
        vosk_model = vosk.Model(self.kws._model_path)
        self.kws.share_model(vosk_model)
        print(f"[Agent] Vosk KWS: {time.monotonic() - started:.2f}s")

        print("[2/4] ASR (Zipformer)...")
        started = time.monotonic()
        self.asr.load()
        print(f"[Agent] ASR load: {time.monotonic() - started:.2f}s")
        print("[3/4] TTS (VITS-Melo)...")
        started = time.monotonic()
        self.tts.load()
        print(f"[Agent] TTS load: {time.monotonic() - started:.2f}s")
        print("[4/4] LLM (NPU)...")
        started = time.monotonic()
        self._load_llm()
        print(f"[Agent] LLM load: {time.monotonic() - started:.2f}s")

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
        if self._shutting_down.is_set():
            return
        self._shutting_down.set()
        print("[Agent] Shutting down...")
        self._busy.set()
        self.kws.stop()
        if (self._interaction_thread and self._interaction_thread.is_alive() and
                self._interaction_thread is not threading.current_thread()):
            self._interaction_thread.join(timeout=20.0)
        if self.llm:
            self.llm.cleanup()
            self.llm = None
        self._stop_event.set()
        print("[Agent] Goodbye.")

    def request_stop(self):
        """Ask the main loop to stop from a signal handler."""
        self._stop_event.set()

    # ----------------------------------------------------------------
    #  Wake callback (called from KWS audio thread — must be fast)
    # ----------------------------------------------------------------
    def _on_wake(self):
        """Handle wake word detection."""
        with self._wake_lock:
            if self._busy.is_set():
                return
            self._busy.set()
        self._interaction_thread = threading.Thread(
            target=self._handle_interaction,
            daemon=True,
        )
        self._interaction_thread.start()

    # ----------------------------------------------------------------
    #  Interaction flow
    # ----------------------------------------------------------------
    def _handle_interaction(self):
        """Full interaction: wake → play "我在" → ASR → LLM → TTS → restart KWS."""
        interaction_started = time.monotonic()
        # Keep the KWS capture open while the confirmation is played. Closing
        # and reopening an HFP source drops the first words of the command.
        # _busy suppresses any duplicate wake callbacks during this interval.

        try:
            # 1. Play pre-generated "我在"
            print("[Agent] Wake! Playing 'wo zai'...")
            self._play_wozai()

            # 2. ASR recording
            started = time.monotonic()
            # VAD still ends ordinary commands promptly; this only prevents
            # long natural questions from being cut at eight seconds.
            question = self.asr.listen_and_transcribe(timeout=25.0)
            print(f"[Agent] ASR stage: {time.monotonic() - started:.2f}s")

            if not question:
                self._speak("我没有听清，请再说一遍")
                return

            print(f"[Agent] User: {question}")

            # LLM inference (with knowledge base)
            started = time.monotonic()
            answer, streamed_speech = self._ask_llm(question)
            print(f"[Agent] LLM stage: {time.monotonic() - started:.2f}s")
            if not answer:
                self._speak("我想不出答案")
                return

            print(f"[Agent] LLM: {answer}")

            # 5. TTS + Play
            if streamed_speech:
                streamed_speech.wait()
            else:
                self._speak(answer)

            # EEG emotion, rather than words in the transcript, controls music.
            if self._consume_music_request():
                time.sleep(0.3)
                self._speak("我检测到你现在情绪不太好，为你播放舒缓音乐。")
                music.play_music(duration=15)

        except Exception as e:
            print(f"[Agent] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[Agent] Interaction total: {time.monotonic() - interaction_started:.2f}s")
            if not self._shutting_down.is_set() and not self._stop_event.is_set():
                # The active KWS stream marks itself detected after a wake.
                # Recreate it after the reply so the next wake can fire, while
                # keeping it alive throughout ASR to avoid HFP cold-start loss.
                self.kws.stop()
                self.kws.start(on_wake=self._on_wake)
            self._busy.clear()

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

    def _ask_llm(self, question: str):
        bci_info = self._get_bci_prompt_context()
        vision_info = self._get_vision_context(question) if self._is_visual_question(question) else ""

        # 检索相关知识库内容
        matches = self.kb.search(question, top_k=2)
        if matches and not vision_info:
            title, content = matches[0]
            steps = [
                re.sub(r"^\d+\.\s*", "", line.strip())
                for line in content.splitlines()
                if re.match(r"^\d+\.", line.strip())
            ]
            if steps:
                print(f"[Agent] KB direct answer: {title}")
                return self._prepare_voice_text(f"{title}：{steps[0]}"), None

        if not self.llm or not self.llm.is_ready():
            return "语言模型未就绪", None

        # Attach only tiny, fresh state snapshots. No request history is kept.
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

        voice_instruction = (
            "请直接用自然口语回答，不要使用Markdown或列表。回答应简洁但完整；"
            "内容较长时用完整短句分段表达，不能在句子中间戛然而止。"
            "涉及矿山事故或急救时优先遵循参考知识；不确定时要求立即报警并等待专业救援，"
            "禁止建议拔出贯穿身体的异物或执行其他高风险操作。"
            "语音转写可能有误；若问题含义不清，先简短确认用户想问什么，不要按猜测作答。"
            "没有提供实时天气、位置或传感器数据时，明确说明无法获取该数据，绝不编造结论。"
        )

        context_parts = [part for part in (bci_info, vision_info, sensor_info) if part]
        if context_parts:
            context = "\n\n".join(context_parts)
            prompt = (
                f"请参考以下信息回答问题，回答要专业、简洁、实用：\n\n"
                f"{context}\n\n"
                f"用户问题：{question}\n{voice_instruction}"
            )
        else:
            prompt = f"{question}\n{voice_instruction}"

        speaker = SentenceSpeechStream(self)
        try:
            answer = self.llm.execute(prompt, max_tokens=256, on_chunk=speaker.on_chunk)
        except Exception:
            speaker._text_queue.put(None)
            speaker.wait()
            raise
        streamed = speaker.finish()
        return self._prepare_voice_text(answer), speaker if streamed else None

    @staticmethod
    def _is_visual_question(question: str) -> bool:
        visual_phrases = (
            "摄像头", "相机", "画面", "图像", "镜头", "看一下前面", "看看前面",
            "前面有什么", "周围有什么", "眼前有什么", "这是什么东西",
        )
        return any(phrase in question for phrase in visual_phrases)

    @staticmethod
    def _get_bci_prompt_context() -> str:
        state = read_context()
        parts = []
        attention = state.get("attention")
        if isinstance(attention, (int, float)):
            label = "专注" if attention >= 0.7 else ("分心" if attention < 0.3 else "一般")
            parts.append(f"注意力{float(attention):.2f}（{label}）")
        emotion = state.get("emotion")
        if isinstance(emotion, dict):
            values = {name: float(emotion.get(name, 0.0)) for name in ("negative", "neutral", "positive")}
            name = max(values, key=values.get)
            labels = {"negative": "偏消极", "neutral": "中性", "positive": "偏积极"}
            parts.append(f"情绪{labels[name]}（{values[name]:.2f}）")
        if not parts:
            return ""
        return "【实时脑电状态】" + "，".join(parts) + "。仅据此调整语气与关怀程度，不把它当作事实判断。"

    def _monitor_bci_emotion(self) -> None:
        """Track fresh EEG emotion updates without adding work to a voice turn."""
        while not self._stop_event.wait(1.0):
            state = read_context()
            emotion_at = state.get("emotion_at")
            emotion = state.get("emotion")
            if not isinstance(emotion_at, (int, float)) or emotion_at <= self._last_emotion_at:
                continue
            self._last_emotion_at = emotion_at
            if not isinstance(emotion, dict):
                continue
            negative = float(emotion.get("negative", 0.0))
            positive = float(emotion.get("positive", 0.0))
            if negative >= 0.65 and negative > positive:
                self._negative_emotion_streak += 1
            else:
                self._negative_emotion_streak = 0
            if self._negative_emotion_streak >= 3 and time.monotonic() - self._last_music_at >= 600:
                with self._music_lock:
                    self._music_requested = True
                self._negative_emotion_streak = 0
                self._last_music_at = time.monotonic()
                print("[Agent] Sustained negative EEG emotion: music requested")

    def _consume_music_request(self) -> bool:
        with self._music_lock:
            requested = self._music_requested
            self._music_requested = False
            return requested

    @staticmethod
    def _get_vision_context(question: str) -> str:
        """Run YOLO in a disposable child process, never in the normal voice path."""
        command = [sys.executable, "-m", "voice_assistant.vision_probe"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
            lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
            data = json.loads(lines[-1]) if lines else {}
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            print(f"[Vision] unavailable: {exc}")
            return "【视觉】摄像头或视觉识别暂不可用，请如实说明无法看到画面。"
        if not data.get("ok"):
            print(f"[Vision] failed: {data.get('error', 'unknown error')}")
            return "【视觉】摄像头或视觉识别暂不可用，请如实说明无法看到画面。"
        objects = data.get("objects", [])
        if not objects:
            return "【视觉 YOLO】当前画面未识别到可确认的物体。"
        summary = "，".join(f"{item['label']} {item['confidence']:.0%}" for item in objects)
        return f"【视觉 YOLO 当前画面】{summary}。请结合用户问题回答，不能编造画面中未检测到的内容。"

    @staticmethod
    def _prepare_voice_text(text: str) -> str:
        """Normalize model output for short, clean speech synthesis."""
        text = re.sub(r"[`*_#>]", "", text or "")
        text = re.sub(r"\s*\n+\s*", "，", text)
        text = re.sub(r"\s+", " ", text).strip(" ，")
        if MAX_VOICE_RESPONSE_CHARS is None or len(text) <= MAX_VOICE_RESPONSE_CHARS:
            return text

        shortened = text[:MAX_VOICE_RESPONSE_CHARS]
        cut = max(shortened.rfind(mark) for mark in "。！？；")
        if cut >= MAX_VOICE_RESPONSE_CHARS // 2:
            return shortened[:cut + 1]
        return shortened.rstrip("，,；;：:") + "。"

    def _speak(self, text: str):
        """Synthesize and play TTS audio."""
        print(f"[TTS] '{text}'")
        cached_audio = self.tts.get_cached_audio(text)
        if cached_audio is not None:
            self._play_audio(cached_audio)
            return

        segments = self._split_tts_segments(text)
        if len(segments) > 1:
            self._play_audio_streaming(segments)
            return

        started = time.monotonic()
        audio = self.tts.synthesize(text)
        print(f"[TTS] synth: {time.monotonic() - started:.2f}s")
        self._play_audio(audio)

    @staticmethod
    def _split_tts_segments(text: str) -> list[str]:
        """Split only at natural speech pauses; text is never omitted."""
        punctuation = "".join(
            chr(code) for code in (0xFF0C, 0x3002, 0xFF01, 0xFF1F, 0xFF1B, 0xFF1A)
        )
        segments = re.findall(f"[^{punctuation}]+[{punctuation}]?", text)
        return segments if len(segments) > 1 else [text]

    @staticmethod
    def _uses_hfp_output() -> bool:
        """HFP needs a short silent warm-up or it drops the first syllables."""
        import subprocess

        try:
            result = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True,
                text=True,
                timeout=0.25,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        sink = result.stdout.strip()
        return sink.startswith("bluez_sink.") and ".handsfree_" in sink

    @staticmethod
    def _to_playback_pcm(audio: np.ndarray) -> np.ndarray:
        """Normalize quiet TTS output before converting it to signed PCM."""
        audio_out = np.clip(audio.astype(np.float32).ravel(), -1.0, 1.0)
        peak = float(np.abs(audio_out).max()) if audio_out.size else 0.0
        if 0 < peak < 0.3:
            audio_out = audio_out * (0.95 / peak)
        return (np.clip(audio_out, -1.0, 1.0) * 32767).astype(np.int16)

    @staticmethod
    def _has_active_capture() -> bool:
        """The wake-word capture keeps the HFP transport warm between replies."""
        import subprocess

        try:
            result = subprocess.run(
                ["pactl", "list", "short", "source-outputs"],
                capture_output=True,
                text=True,
                timeout=0.25,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return bool(result.stdout.strip())

    def _hfp_preroll_pcm(self) -> np.ndarray:
        if not self._uses_hfp_output():
            return np.array([], dtype=np.int16)
        seconds = 0.18 if self._has_active_capture() else 0.55
        return np.zeros(int(self.tts.sample_rate * seconds), dtype=np.int16)

    def _hfp_tail_pcm(self) -> np.ndarray:
        """Keep HFP open briefly so the final syllable is not clipped."""
        if not self._uses_hfp_output():
            return np.array([], dtype=np.int16)
        return np.zeros(int(self.tts.sample_rate * 0.18), dtype=np.int16)

    def _play_audio_streaming(self, segments: list[str]):
        """Generate the next sentence while ALSA plays the current one."""
        import subprocess

        synth_started = time.monotonic()
        audio = self.tts.synthesize(segments[0])
        print(f"[TTS] first segment synth: {time.monotonic() - synth_started:.2f}s")
        if audio is None or audio.size == 0:
            return

        process = None
        audio_started = None
        total_duration = 0.0
        try:
            process = subprocess.Popen(
                [
                    "aplay", "-q", "-t", "raw", "-f", "S16_LE",
                    "-c", "1", "-r", str(self.tts.sample_rate),
                    "--buffer-time=60000", "--period-time=15000",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            preroll = self._hfp_preroll_pcm()
            if preroll.size:
                audio_started = time.monotonic()
                process.stdin.write(preroll.tobytes())
                process.stdin.flush()
                total_duration += len(preroll) / self.tts.sample_rate
            for index, segment in enumerate(segments):
                if audio is not None and audio.size:
                    audio_i16 = self._to_playback_pcm(audio)
                    if audio_started is None:
                        audio_started = time.monotonic()
                    process.stdin.write(audio_i16.tobytes())
                    process.stdin.flush()
                    total_duration += len(audio_i16) / self.tts.sample_rate

                if index + 1 < len(segments):
                    synth_started = time.monotonic()
                    audio = self.tts.synthesize(segments[index + 1])
                    print(
                        f"[TTS] segment {index + 2} synth: "
                        f"{time.monotonic() - synth_started:.2f}s"
                    )

            tail = self._hfp_tail_pcm()
            if tail.size:
                process.stdin.write(tail.tobytes())
                total_duration += len(tail) / self.tts.sample_rate
            process.stdin.close()
            elapsed = time.monotonic() - (audio_started or time.monotonic())
            process.wait(timeout=max(0.5, total_duration - elapsed + 0.75))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)
        except (BrokenPipeError, OSError, subprocess.SubprocessError):
            if process and process.poll() is None:
                process.kill()

    def _play_wozai(self):
        """Play pre-generated '我在' audio."""
        audio = self.tts.wozai_audio
        self._play_audio(audio)

    def _play_audio(self, audio: np.ndarray):
        """Stream raw PCM to ALSA without a temporary WAV file."""
        if audio is None or audio.size == 0:
            return
        import subprocess
        audio_i16 = self._to_playback_pcm(audio)
        preroll = self._hfp_preroll_pcm()
        tail = self._hfp_tail_pcm()
        payload = preroll.tobytes() + audio_i16.tobytes() + tail.tobytes()
        duration = (len(preroll) + len(audio_i16) + len(tail)) / self.tts.sample_rate
        process = None
        try:
            process = subprocess.Popen(
                [
                    "aplay", "-q", "-t", "raw", "-f", "S16_LE",
                    "-c", "1", "-r", str(self.tts.sample_rate),
                    "--buffer-time=60000", "--period-time=15000",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            process.communicate(
                input=payload,
                timeout=duration + 0.5,
            )
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)
        except (OSError, subprocess.SubprocessError):
            if process and process.poll() is None:
                process.kill()

    # ----------------------------------------------------------------
    #  Interactive mode
    # ----------------------------------------------------------------
    def run_forever(self):
        """Blocking main loop. Press Ctrl+C to stop."""
        self.start()
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            print("\n[Agent] Interrupted by user.")
        finally:
            self.stop()


def main():
    agent = VoiceAgent()
    signal.signal(signal.SIGTERM, lambda signum, frame: agent.request_stop())
    signal.signal(signal.SIGINT, lambda signum, frame: agent.request_stop())
    agent.run_forever()


if __name__ == "__main__":
    main()

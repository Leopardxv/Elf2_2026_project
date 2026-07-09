#!/usr/bin/env python3
"""
ELF2 Modular Agent
==================
Architecture:
    ┌──────────────────────────────────────────┐
    │  Background Monitors (running constantly) │
    │  EEG Attention ──┐                        │
    │  EEG Emotion  ───┼──→ Context State      │
    └──────────────────┼───────────────────────┘
                       │
    ┌──────────────────┴───────────────────────┐
    │  Interactive Modules                     │
    │                                          │
    │  User Question → Router →                │
    │    ├── vision? → VisionModule (YOLO)     │
    │    └── always → LLMModule               │
    │                  (with context prompt)    │
    └──────────────────────────────────────────┘

Usage:
    agent = ModularAgent()
    agent.start()           # load models + start EEG monitors
    answer = agent.ask("Is there danger ahead?")
    agent.stop()
"""
import time, os
from .context import ctx
from .llm import LLMModule
from .vision import VisionModule
from .eeg_attention import AttentionMonitor
from .eeg_emotion import EmotionMonitor


class ModularAgent:
    def __init__(self):
        self.llm = LLMModule()
        self.vision = VisionModule(camera_id=21)
        self.attention_mon = AttentionMonitor()
        self.emotion_mon = EmotionMonitor()
        self._started = False

    # ================================================================
    #  Lifecycle
    # ================================================================
    def start(self, enable_eeg=True):
        """Load all models and start monitors. Call once at startup."""

        # Set NPU frequency
        os.system("echo elf | sudo -S bash -c \"echo userspace > /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null; echo 1000000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq 2>/dev/null\"")
        print("NPU: 1GHz")

        # Load LLM
        print("\n[1/4] LLM (Qwen2.5 NPU)...")
        self.llm.load()

        # Load Vision
        print("\n[2/4] Vision (YOLOv5s NPU)...")
        self.vision.load()

        # Load & start EEG Emotion
        print("\n[3/4] EEG Emotion (Conformer NPU)...")
        if enable_eeg:
            self.emotion_mon.load()
            self.emotion_mon.start()

        # Start EEG Attention (UDP listener)
        print("\n[4/4] EEG Attention (BrainFlow UDP)...")
        if enable_eeg:
            self.attention_mon.start()

        # Set initial context for demo (no EEG hardware)
        ctx.set_attention(0.4)
        ctx.set_emotion("neutral", 0.8)

        self._started = True
        print("\n" + "=" * 55)
        print("Agent ready.")
        print("  Modules: [LLM] [Vision] [Attention] [Emotion]")
        print("  Context: attention={:.1f} emotion={}".format(
            ctx.attention, ctx.emotion))
        print("=" * 55)

    def stop(self):
        self.attention_mon.stop()
        self.emotion_mon.stop()
        self.llm.cleanup()
        self.vision.cleanup()
        self.emotion_mon.cleanup()
        self._started = False

    # ================================================================
    #  Main API
    # ================================================================
    def ask(self, question: str) -> str:
        """Ask a question. Agent routes to vision if needed, then LLM with context."""
        if not self._started:
            return "Agent not started. Call agent.start() first."

        print("\n[Agent] Q: {}".format(question))
        print("[Agent] Context: attention={:.1f} emotion={}".format(
            ctx.attention, ctx.emotion))

        # Step 1: Route
        needs_vision = self._route(question)

        # Step 2: Execute vision if needed
        vision_context = ""
        if needs_vision:
            print("[Agent] → Vision module...")
            objects = self.vision.detect()
            if objects:
                top = sorted(objects, key=lambda x: x["confidence"], reverse=True)[:8]
                obj_list = ", ".join("{} ({:.0%})".format(o["label"], o["confidence"]) for o in top)
                vision_context = "Camera detected: {}. ".format(obj_list)
                print("[Agent] Detected: {}".format(obj_list[:100]))
            else:
                vision_context = "Camera showed nothing notable. "
                print("[Agent] No objects detected")

        # Step 3: LLM with context
        print("[Agent] → LLM (with EEG context)...")
        t0 = time.time()
        full_question = vision_context + question
        answer = self.llm.ask(full_question)
        elapsed = time.time() - t0
        print("[Agent] Time: {:.1f}s".format(elapsed))
        return answer

    # ================================================================
    #  Router
    # ================================================================
    def _route(self, question: str) -> bool:
        """Return True if vision (YOLO) should be used."""
        q = question.lower()

        vision_keywords = [
            "what do you see", "what can you see", "what is in front",
            "describe the room", "describe the scene", "describe what",
            "is there a", "is there an", "are there any",
            "how many people", "how many cars", "how many objects",
            "what color is the", "what object", "what objects",
            "take a picture", "take a photo", "show me",
            "look at", "check the camera",
            "看看", "看一下", "摄像头", "前面有", "有什么东西",
            "几个", "什么样",
            "see", "look", "camera", "image", "picture", "photo",
        ]

        for kw in vision_keywords:
            if kw in q:
                return True
        return False

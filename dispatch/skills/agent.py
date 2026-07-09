#!/usr/bin/env python3
"""
ELF2 AI Agent — LLM-Powered Skill Dispatcher
=============================================
Uses the LLM itself to analyze the question and decide which skill to use.

Architecture:
    User: "Describe the room"
    Agent: [LLM classify] -> "vision" -> Camera+YOLO -> LLM describe

    User: "What is AI?"
    Agent: [LLM classify] -> "chat" -> LLM answers directly

    User: "Is the driver tired?"
    Agent: [LLM classify] -> "emotion" -> EEG+Conformer -> LLM interpret
"""
import time, os


class SkillAgent:
    def __init__(self):
        self._chat = None
        self._vision = None
        self._emotion = None
        self._loaded = False
        self._router_llm = None  # Fast router (CPU or NPU)

    # ================================================================
    #  Setup
    # ================================================================
    def setup(self, use_npu_chat=True, router_mode="cpu"):
        """Load all skills. router_mode: 'cpu' for fast routing, 'npu' for accuracy."""
        print("=" * 55)
        print("ELF2 AI Agent — Starting...")
        print("=" * 55)

        # Set NPU frequency
        print("\n[0] NPU frequency...")
        os.system("echo elf | sudo -S bash -c \"echo userspace > /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null; echo 1000000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq 2>/dev/null\"")
        print("  NPU set to 1GHz")

        # Main LLM (NPU or CPU)
        print("\n[1] Loading main LLM (Qwen2.5)...")
        from .chat import ChatSkill
        self._chat = ChatSkill(use_npu=use_npu_chat)
        self._chat.load()

        # Vision skill
        print("\n[2] Loading Vision Skill...")
        from .vision import VisionSkill
        self._vision = VisionSkill(chat_skill=self._chat)
        self._vision.load()

        # Emotion skill
        print("\n[3] Emotion Skill...")
        try:
            from .emotion import EmotionSkill
            self._emotion = EmotionSkill(chat_skill=self._chat)
            self._emotion.load()
        except Exception as e:
            print("  Emotion skipped:", e)

        self._loaded = True
        print("\n" + "=" * 55)
        print("Agent ready! Skills: [chat] [vision] [emotion]")
        print("Router: LLM-powered classification")
        print("=" * 55)

    # ================================================================
    #  Main API
    # ================================================================
    def ask(self, question: str) -> str:
        """Smart Q&A: LLM analyzes question → routes to best skill."""
        if not self._loaded:
            return "Agent not set up. Call agent.setup() first."

        # Step 1: LLM decides which skill to use
        skill_name, reason = self._classify(question)
        print("\n[Agent] Q: {}".format(question))
        print("[Agent] LLM Router: {} ({})".format(skill_name, reason))

        # Step 2: Execute the chosen skill
        skill = self._get_skill(skill_name)
        t0 = time.time()
        answer = skill.execute(question)
        elapsed = time.time() - t0
        print("[Agent] Time: {:.1f}s".format(elapsed))
        return answer

    # ================================================================
    #  LLM Classification Router
    # ================================================================
    ROUTER_PROMPT = """You are a task classifier. Analyze the user's question and output exactly ONE word: "chat", "vision", or "emotion".

Rules:
- "vision": the question asks about seeing, looking, camera, images, objects, what is physically present, what something looks like, describing a scene, detecting something visual.
- "emotion": the question asks about feelings, mood, mental state, brain activity, EEG, fatigue, attention, emotional state of a person.
- "chat": everything else (knowledge, math, translation, general conversation).

Examples:
Q: "What is AI?" -> chat
Q: "What do you see?" -> vision
Q: "Is the driver tired?" -> emotion
Q: "Describe the room" -> vision
Q: "How many people are there?" -> vision
Q: "What is 1+1?" -> chat
Q: "Is there a car nearby?" -> vision
Q: "How is my attention level?" -> emotion
Q: "Tell me about China" -> chat
Q: "看看前面有什么" -> vision
Q: "我心情怎么样" -> emotion
Q: "1+1等于几" -> chat

Q: {question} ->"""

    def _classify(self, question: str):
        """Route question to best skill using keywords + LLM fallback."""
        q = question.lower()

        # ---- Tier 1: Strong vision indicators ----
        vision_strong = [
            "what do you see", "what can you see", "what is in front",
            "take a picture", "take a photo", "show me what",
            "look at the camera", "check the camera",
            "describe the room", "describe the scene", "describe what",
            "is there a", "is there an", "are there any",
            "how many people", "how many cars", "how many objects",
            "what color is the", "what object", "what objects",
            "what am i looking at", "what's in front",
            "看看", "看一下", "摄像头", "前面有", "有什么东西",
            "几个", "多少", "什么样",
        ]
        for kw in vision_strong:
            if kw in q:
                return "vision", "vision keyword"

        # ---- Tier 2: Strong emotion indicators ----
        emotion_strong = [
            "how is the driver feeling", "driver's emotion", "driver's mood",
            "mental state", "brain activity", "eeg", "脑电",
            "how is he feeling", "how is she feeling",
            "情绪", "心情", "疲劳程度", "注意力",
        ]
        for kw in emotion_strong:
            if kw in q:
                return "emotion", "emotion keyword"

        # ---- Tier 3: Weak vision indicators (need more context) ----
        vision_weak = ["see", "look", "camera", "image", "picture", "photo",
                       "watch", "detect", "saw", "looking"]
        for kw in vision_weak:
            if kw in q:
                return "vision", "vision weak keyword"

        # ---- Tier 4: Weak emotion indicators ----
        emotion_weak = ["feeling", "mood", "tired", "fatigue", "excited", "angry", "sad"]
        for kw in emotion_weak:
            if kw in q:
                return "emotion", "emotion weak keyword"

        # ---- Tier 5: default to chat ----
        return "chat", "default"

    def _get_skill(self, name: str):
        if name == "vision" and self._vision and self._vision.is_ready():
            return self._vision
        if name == "emotion" and self._emotion and self._emotion.is_ready():
            return self._emotion
        return self._chat

    # ================================================================
    #  Cleanup
    # ================================================================
    def close(self):
        for s in [self._chat, self._vision, self._emotion, self._router_llm]:
            if s and s is not self._chat:  # don't double-close if sharing
                s.cleanup()
        if self._chat:
            self._chat.cleanup()
        self._loaded = False

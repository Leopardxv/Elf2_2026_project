#!/usr/bin/env python3
import os

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# NPU LLM (RKLLM - needs NPU driver >= 0.9.7, currently segfault bug)
# DEEPSEEK_RKLLM = os.path.join(_MODEL_DIR, "DeepSeek-R1-Distill-Qwen-1.5B_W8A8_RK3588.rkllm")

# CPU LLM (llama-cpp-python GGUF) - 396MB, works!
LLAMA_MODEL = os.path.join(_MODEL_DIR, "qwen2.5-0.5b-instruct-q2_k.gguf")

# NPU vision models
RESNET18_RKNN = os.path.join(_MODEL_DIR, "resnet18_for_rk3588.rknn")
YOLOV5S_RKNN = os.path.join(_MODEL_DIR, "yolov5s-640-640.rknn")

# NPU core count
NPU_CORE_COUNT = 3

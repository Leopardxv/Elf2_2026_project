# dispatch - safe imports
try:
    from .llama_engine import LlamaEngine
except Exception:
    pass
try:
    from .rknn_vision import VisionNPU
except Exception:
    pass
try:
    from .rkllm_engine import RKLLMEngine
except Exception:
    pass

#!/usr/bin/env python3
"""
RKLLM Engine — Python ctypes wrapper for librkllmrt.so

Supports:
- LLM (text-only): DeepSeek-R1-1.5B, Qwen, TinyLLAMA, etc.
- VLM (multimodal): Qwen2-VL, Qwen3-VL, InternVL

Usage:
    engine = RKLLMEngine()
    engine.init("/path/to/model.rkllm")
    engine.run("你好", on_token=lambda t: print(t, end=""))
    engine.destroy()
"""
import ctypes
import os
import sys
from ctypes import (c_char_p, c_uint8, c_int32, c_int8, c_uint32, c_uint16,
                    c_float, c_bool, c_void_p, c_size_t, POINTER, Structure,
                    Union, CFUNCTYPE, byref, cast, addressof)
from typing import Callable, Optional

# ---- Enums ----
LLMCallState = c_int32
RKLLMInputType = c_int32
RKLLMInferMode = c_int32

# LLMCallState
RKLLM_RUN_NORMAL = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH = 2
RKLLM_RUN_ERROR = 3
RKLLM_RUN_GET_LAST_HIDDEN_LAYER = 4

# RKLLMInputType
RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_TOKEN = 1
RKLLM_INPUT_EMBED = 2
RKLLM_INPUT_MULTIMODAL = 3

# RKLLMInferMode
RKLLM_INFER_GENERATE = 0
RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1

# ---- Structs ----
class RKLLMExtendParam(Structure):
    _fields_ = [
        ("base_domain_id", c_int32),
        ("reserved", c_uint8 * 112),
    ]

class RKLLMParam(Structure):
    _fields_ = [
        ("model_path", c_char_p),
        ("max_context_len", c_int32),
        ("max_new_tokens", c_int32),
        ("top_k", c_int32),
        ("top_p", c_float),
        ("temperature", c_float),
        ("repeat_penalty", c_float),
        ("frequency_penalty", c_float),
        ("presence_penalty", c_float),
        ("mirostat", c_int32),
        ("mirostat_tau", c_float),
        ("mirostat_eta", c_float),
        ("skip_special_token", c_bool),
        ("is_async", c_bool),
        ("img_start", c_char_p),
        ("img_end", c_char_p),
        ("img_content", c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMLoraAdapter(Structure):
    _fields_ = [
        ("lora_adapter_path", c_char_p),
        ("lora_adapter_name", c_char_p),
        ("scale", c_float),
    ]

class RKLLMEmbedInput(Structure):
    _fields_ = [
        ("embed", POINTER(c_float)),
        ("n_tokens", c_size_t),
    ]

class RKLLMTokenInput(Structure):
    _fields_ = [
        ("input_ids", POINTER(c_int32)),
        ("n_tokens", c_size_t),
    ]

class RKLLMMultiModelInput(Structure):
    _fields_ = [
        ("prompt", c_char_p),
        ("image_embed", POINTER(c_float)),
        ("n_image_tokens", c_size_t),
    ]

class _RKLLMInputUnion(Union):
    _fields_ = [
        ("prompt_input", c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput),
    ]

class RKLLMInput(Structure):
    _anonymous_ = ("_u",)
    _fields_ = [
        ("input_type", RKLLMInputType),
        ("_u", _RKLLMInputUnion),
    ]

class RKLLMLoraParam(Structure):
    _fields_ = [
        ("lora_adapter_name", c_char_p),
    ]

class RKLLMPromptCacheParam(Structure):
    _fields_ = [
        ("save_prompt_cache", c_int32),
        ("prompt_cache_path", c_char_p),
    ]

class RKLLMInferParam(Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", POINTER(RKLLMPromptCacheParam)),
    ]

class RKLLMResultLastHiddenLayer(Structure):
    _fields_ = [
        ("hidden_states", POINTER(c_float)),
        ("embd_size", c_int32),
        ("num_tokens", c_int32),
    ]

class RKLLMResult(Structure):
    _fields_ = [
        ("text", c_char_p),
        ("token_id", c_int32),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
    ]

# ---- Callback type ----
LLMResultCallback = CFUNCTYPE(
    None,
    POINTER(RKLLMResult),
    c_void_p,
    LLMCallState,
)

# ---- Main Engine ----
class RKLLMEngine:
    """
    Python wrapper for RKLLM C API.

    Supports both LLM (text) and VLM (multimodal) models.
    Only one model loaded at a time.
    """

    def __init__(self, lib_path: Optional[str] = None):
        """
        Args:
            lib_path: path to librkllmrt.so.
                      If None, searches system paths and project tree.
        """
        self._lib = None
        self._handle: Optional[c_void_p] = None
        self._loaded = False
        self._lib_path = lib_path
        self._on_token: Optional[Callable[[str], None]] = None
        self._callback_ref = None  # keep callback alive

    # ----------------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------------
    def init(self, model_path: str, **kwargs) -> int:
        """
        Load model to NPU. Must call before run().

        Args:
            model_path:  path to .rkllm model file
            max_context_len: max context tokens (default 1024)
            max_new_tokens:  max tokens to generate (default 512)
            temperature:     sampling temperature (default 0.8)
            top_p:           nucleus sampling (default 0.95)
            top_k:           top-k sampling (default 1)
            skip_special:    skip special tokens (default True)
        Returns:
            0 on success, non-0 on failure.
        """
        if self._loaded:
            print("[RKLLM] Model already loaded, call destroy() first")
            return -1

        self._load_library()

        if not os.path.isfile(model_path):
            print(f"[RKLLM] Model file not found: {model_path}")
            return -1

        # Build param struct
        param = RKLLMParam()
        self._init_param_defaults(param)

        model_path_bytes = model_path.encode("utf-8")
        param.model_path = c_char_p(model_path_bytes)

        param.max_context_len = kwargs.get("max_context_len", 4096)
        param.max_new_tokens = kwargs.get("max_new_tokens", 512)
        param.top_k = kwargs.get("top_k", 1)
        param.top_p = kwargs.get("top_p", 0.95)
        param.temperature = kwargs.get("temperature", 0.8)
        param.repeat_penalty = kwargs.get("repeat_penalty", 1.1)
        param.frequency_penalty = kwargs.get("frequency_penalty", 0.0)
        param.presence_penalty = kwargs.get("presence_penalty", 0.0)
        param.skip_special_token = kwargs.get("skip_special", True)
        param.is_async = False
        param.extend_param.base_domain_id = 0

        # Set multimodal markers (used for VLM models)
        img_start = kwargs.get("img_start", "")
        img_end = kwargs.get("img_end", "")
        img_content = kwargs.get("img_content", "")
        if img_start:
            param.img_start = img_start.encode("utf-8")
        if img_end:
            param.img_end = img_end.encode("utf-8")
        if img_content:
            param.img_content = img_content.encode("utf-8")

        # Init handle — callback is required, create a permanent one
        self._callback_ref = LLMResultCallback(self._token_callback)
        handle = c_void_p()
        ret = self._lib.rkllm_init(byref(handle), byref(param), self._callback_ref)
        if ret != 0:
            self._callback_ref = None
            print(f"[RKLLM] init failed: code={ret}")
            return ret

        self._handle = handle
        self._loaded = True
        print(f"[RKLLM] Model loaded: {model_path}")
        return 0

    def run(self, prompt: str,
            on_token: Optional[Callable[[str], None]] = None,
            image_embed_data: Optional[bytes] = None,
            n_image_tokens: int = 0,
            max_new_tokens: int = -1) -> str:
        """
        Run inference with streaming callbacks.

        Args:
            prompt:           input text
            on_token:         callback(token_text) for each generated token
            image_embed_data: VLM: raw float32 image embedding bytes
            n_image_tokens:   VLM: number of image tokens
            max_new_tokens:   override default (-1 = use init value)
        Returns:
            Full generated text.
        """
        if not self._loaded:
            raise RuntimeError("[RKLLM] Model not loaded. Call init() first.")

        self._on_token = on_token
        self._output_buf = []

        rkllm_input = RKLLMInput()
        prompt_bytes = prompt.encode("utf-8")

        if image_embed_data is not None and len(image_embed_data) > 0:
            rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL
            rkllm_input.multimodal_input.prompt = prompt_bytes
            rkllm_input.multimodal_input.image_embed = cast(
                image_embed_data, POINTER(c_float))
            rkllm_input.multimodal_input.n_image_tokens = n_image_tokens
        else:
            rkllm_input.input_type = RKLLM_INPUT_PROMPT
            rkllm_input.prompt_input = prompt_bytes

        infer_param = RKLLMInferParam()
        infer_param.mode = RKLLM_INFER_GENERATE

        ret = self._lib.rkllm_run(
            self._handle,
            byref(rkllm_input),
            byref(infer_param),
            None,
        )

        self._on_token = None

        if ret != 0:
            print(f"[RKLLM] run failed: code={ret}")
            return ""

        return "".join(self._output_buf)

    def run_multimodal(self, prompt: str, image_embed_data: bytes,
                       n_image_tokens: int = 0,
                       on_token: Optional[Callable[[str], None]] = None) -> str:
        return self.run(prompt, on_token=on_token,
                        image_embed_data=image_embed_data,
                        n_image_tokens=n_image_tokens)

    def destroy(self):
        """Unload model from NPU and release resources."""
        if self._handle and self._loaded:
            self._lib.rkllm_destroy(self._handle)
            self._handle = None
            self._loaded = False
            print("[RKLLM] Model unloaded.")

    def is_loaded(self) -> bool:
        return self._loaded

    # ----------------------------------------------------------------
    #  Internal
    # ----------------------------------------------------------------
    def _load_library(self):
        if self._lib is not None:
            return

        paths = []
        if self._lib_path:
            paths.append(self._lib_path)
        # project tree
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paths.append(os.path.join(project_root, "ai", "deepseek", "rknn-llm-main",
                                  "rkllm-runtime", "Linux", "librkllm_api",
                                  "aarch64", "librkllmrt.so"))
        # system paths
        paths.append("librkllmrt.so")

        for p in paths:
            try:
                self._lib = ctypes.CDLL(p)
                print(f"[RKLLM] Loaded library: {p}")
                break
            except OSError:
                continue

        if self._lib is None:
            raise RuntimeError(
                "[RKLLM] Cannot find librkllmrt.so. "
                "Ensure it's installed in /usr/local/lib/ or set lib_path."
            )

        # ---- Set function signatures ----
        # rkllm_createDefaultParam
        self._lib.rkllm_createDefaultParam.restype = RKLLMParam

        # rkllm_init
        self._lib.rkllm_init.argtypes = [
            POINTER(c_void_p),          # LLMHandle*
            POINTER(RKLLMParam),        # param
            LLMResultCallback,          # callback
        ]
        self._lib.rkllm_init.restype = c_int32

        # rkllm_run
        self._lib.rkllm_run.argtypes = [
            c_void_p,                   # handle
            POINTER(RKLLMInput),        # input
            POINTER(RKLLMInferParam),   # infer_params
            c_void_p,                   # userdata
        ]
        self._lib.rkllm_run.restype = c_int32

        # rkllm_destroy
        self._lib.rkllm_destroy.argtypes = [c_void_p]
        self._lib.rkllm_destroy.restype = c_int32

        # rkllm_is_running
        self._lib.rkllm_is_running.argtypes = [c_void_p]
        self._lib.rkllm_is_running.restype = c_int32

        # rkllm_abort
        self._lib.rkllm_abort.argtypes = [c_void_p]
        self._lib.rkllm_abort.restype = c_int32

    @staticmethod
    def _init_param_defaults(param: RKLLMParam):
        """Zero-init all fields. rkllm_createDefaultParam is optional."""
        ctypes.memset(byref(param), 0, ctypes.sizeof(param))

    def _token_callback(self, result: POINTER(RKLLMResult),
                        userdata: c_void_p, state: LLMCallState):
        """
        C callback: called by librkllmrt.so during rkllm_run().

        Streams token text to Python callback.
        """
        if state == RKLLM_RUN_FINISH:
            return
        elif state == RKLLM_RUN_ERROR:
            # Don't print empty buffer on error; error is already logged by rkllm_run
            return

        text = result.contents.text
        if text and state in (RKLLM_RUN_NORMAL, RKLLM_RUN_WAITING):
            try:
                s = text.decode("utf-8") if isinstance(text, bytes) else text
            except UnicodeDecodeError:
                return

            self._output_buf.append(s)
            if self._on_token:
                self._on_token(s)


# ---- Convenience ----
def quick_test(model_path: str):
    """Minimal smoke test — requires a .rkllm file."""
    engine = RKLLMEngine()
    ret = engine.init(model_path)
    if ret != 0:
        print(f"Init failed: {ret}")
        return

    resp = engine.run("你好，用一句话介绍自己。")
    print(f"\n=== Response ===\n{resp}")
    engine.destroy()

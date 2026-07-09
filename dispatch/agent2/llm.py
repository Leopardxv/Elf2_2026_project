"""
LLM Module — Qwen2.5 on NPU (or CPU fallback).
Reads Context for system prompt enrichment.
"""
import ctypes, time, os
from ctypes import *
from .context import ctx

# ---- RKLLM structs ----
class _E(Structure): _fields_ = [("a", c_int32), ("b", c_uint8 * 112)]
class _P(Structure):
    _fields_ = [
        ("model_path", c_char_p), ("max_context_len", c_int32), ("max_new_tokens", c_int32),
        ("top_k", c_int32), ("top_p", c_float), ("temperature", c_float),
        ("repeat_penalty", c_float), ("frequency_penalty", c_float), ("presence_penalty", c_float),
        ("mirostat", c_int32), ("mirostat_tau", c_float), ("mirostat_eta", c_float),
        ("skip_special_token", c_bool), ("is_async", c_bool),
        ("img_start", c_char_p), ("img_end", c_char_p), ("img_content", c_char_p),
        ("extend_param", _E),
    ]
class _EI(Structure): _fields_ = [("embed", POINTER(c_float)), ("n_tokens", c_size_t)]
class _TI(Structure): _fields_ = [("input_ids", POINTER(c_int32)), ("n_tokens", c_size_t)]
class _MI(Structure): _fields_ = [("prompt", c_char_p), ("image_embed", POINTER(c_float)), ("n_image_tokens", c_size_t)]
class _U(Union): _fields_ = [("prompt_input", c_char_p), ("embed_input", _EI), ("token_input", _TI), ("multimodal_input", _MI)]
class _INP(Structure): _anonymous_ = ("_u",); _fields_ = [("input_type", c_int32), ("_u", _U)]
class _LP(Structure): _fields_ = [("lora_adapter_name", c_char_p)]
class _PCP(Structure): _fields_ = [("save_prompt_cache", c_int32), ("prompt_cache_path", c_char_p)]
class _IP(Structure): _fields_ = [("mode", c_int32), ("lora_params", POINTER(_LP)), ("prompt_cache_params", POINTER(_PCP))]
class _R(Structure): _fields_ = [("text", c_char_p), ("token_id", c_int32), ("last_hidden_layer", c_void_p)]
_CALLBACK = CFUNCTYPE(None, POINTER(_R), c_void_p, c_int32)

NPU_MODEL = "/home/elf/Projects/models/Qwen2.5-0.5B-Instruct_W8A8_RK3588_CN.rkllm"


class LLMModule:
    def __init__(self):
        self._lib = None
        self._handle = None
        self._cb_fn = None
        self._text_buf = []
        self._ttft = 0.0
        self._first = True
        self._loaded = False

    def load(self) -> bool:
        self._lib = ctypes.CDLL("/usr/local/lib/librkllmrt.so")
        self._lib.rkllm_createDefaultParam.restype = _P
        self._lib.rkllm_init.argtypes = [POINTER(c_void_p), POINTER(_P), _CALLBACK]
        self._lib.rkllm_init.restype = c_int32
        self._lib.rkllm_run.argtypes = [c_void_p, POINTER(_INP), POINTER(_IP), c_void_p]
        self._lib.rkllm_run.restype = c_int32
        self._lib.rkllm_destroy.argtypes = [c_void_p]; self._lib.rkllm_destroy.restype = c_int32

        def cb(r, u, s):
            if s >= 2: return
            now = time.time()
            if self._first: self._ttft = now; self._first = False
            if r and r.contents.text: self._text_buf.append(r.contents.text)
        self._cb_fn = _CALLBACK(cb)

        param = self._lib.rkllm_createDefaultParam()
        param.model_path = NPU_MODEL.encode()
        param.max_context_len = 1024; param.max_new_tokens = 256
        param.top_k = 1; param.top_p = 0.9; param.temperature = 0.7
        param.repeat_penalty = 1.1; param.skip_special_token = True
        h = c_void_p()

        t0 = time.time()
        ret = self._lib.rkllm_init(byref(h), byref(param), self._cb_fn)
        if ret != 0:
            print("[LLM] Init failed:", ret)
            return False
        self._handle = h
        self._loaded = True
        print("[LLM] Loaded ({:.0f}s)".format(time.time() - t0))
        return True

    def ask(self, question: str, system_prompt: str = "", max_tokens: int = 256) -> str:
        """LLM inference with context-aware system prompt."""
        if not self._loaded:
            return "[LLM] Not loaded"

        nl = chr(10)
        im_s, im_e = "<|im_start|>", "<|im_end|>"

        if not system_prompt:
            system_prompt = ctx.get_system_prompt()

        prompt = nl.join([
            im_s + "system", system_prompt + im_e,
            im_s + "user", question + im_e,
            im_s + "assistant", "",
        ])

        self._text_buf = []; self._first = True
        inp = _INP(); inp.input_type = 0; inp.prompt_input = prompt.encode()
        iparam = _IP(); iparam.mode = 0; iparam.lora_params = None; iparam.prompt_cache_params = None
        self._lib.rkllm_run(self._handle, byref(inp), byref(iparam), None)
        raw = b"".join(self._text_buf)
        return raw.decode("utf-8", errors="replace").strip()

    def is_loaded(self) -> bool:
        return self._loaded

    def cleanup(self):
        if self._lib and self._handle:
            self._lib.rkllm_destroy(self._handle)
            self._loaded = False

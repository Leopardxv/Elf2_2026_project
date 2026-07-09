#!/usr/bin/env python3
"""
ELF2 AI Engine — Unified NPU Inference API
===========================================
Usage:
    ai = ELFAI()
    ai.load_yolo()    # load YOLOv5s on NPU
    ai.load_llm()     # load Qwen on NPU  (or cpu_llm=True for CPU)

    objects = ai.detect(image_bgr)
    answer = ai.ask("What do you see?")
    ai.ask_with_vision(image_bgr, "Describe this scene")
"""

import ctypes, sys, time, os, numpy as np
from ctypes import *

# ============================================================
#  RKLLM Ctypes Structs (for NPU LLM)
# ============================================================
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

# ============================================================
#  Main API Class
# ============================================================
class ELFAI:
    # ---- Model paths ----
    YOLO_MODEL = "/home/elf/Projects/models/yolov5s-640-640.rknn"
    NPU_LLM_MODEL = "/home/elf/Projects/models/Qwen2.5-0.5B-Instruct_W8A8_RK3588_CN.rkllm"
    CPU_LLM_MODEL = "/home/elf/Projects/models/qwen2.5-0.5b-instruct-q2_k.gguf"

    COCO_NAMES = [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
        "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
        "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
        "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
        "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
        "potted plant","bed","dining table","toilet","tv","laptop","mouse",
        "remote","keyboard","cell phone","microwave","oven","toaster","sink",
        "refrigerator","book","clock","vase","scissors","teddy bear",
        "hair drier","toothbrush",
    ]

    def __init__(self):
        self._rknn = None          # YOLO RKNNLite instance
        self._llm_handle = None    # RKLLM handle
        self._llm_lib = None       # librkllmrt CDLL
        self._llm_cb_fn = None     # Callback (kept alive)
        self._cpu_llm = None       # CPU LLM instance
        self._llm_mode = None      # "npu" or "cpu"

    # ================================================================
    #  YOLO Object Detection (NPU)
    # ================================================================
    def load_yolo(self, model_path=None):
        """Load YOLOv5s on NPU. Call once at startup."""
        if self._rknn:
            return True
        from rknnlite.api import RKNNLite
        path = model_path or self.YOLO_MODEL
        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(path)
        if ret != 0:
            print("[YOLO] load_rknn failed:", ret)
            return False
        ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print("[YOLO] init_runtime failed:", ret)
            return False
        print("[YOLO] Loaded:", os.path.basename(path))
        return True

    def detect(self, image_bgr, conf_thresh=0.35, iou_thresh=0.45):
        """
        Detect objects in BGR image (numpy array, HxWx3).
        Returns list of dicts: {label, confidence, box: [x1,y1,x2,y2]}
        """
        if not self._rknn:
            raise RuntimeError("[YOLO] Not loaded. Call load_yolo() first.")

        orig_h, orig_w = image_bgr.shape[:2]
        import cv2
        resized = cv2.resize(image_bgr, (640, 640))
        inp = np.transpose(resized, (2, 0, 1)).astype(np.float32)[None, :]
        outputs = self._rknn.inference(inputs=[inp])
        return self._yolov5_postprocess(outputs, orig_w, orig_h, conf_thresh, iou_thresh)

    # ================================================================
    #  LLM Chat (NPU or CPU)
    # ================================================================
    def load_llm(self, model_path=None, cpu=False):
        """
        Load LLM. Set cpu=True for CPU model (smaller, Chinese-capable).
        Set cpu=False for NPU model (faster, English-optimized, CN with CN-calibrated model).
        """
        if cpu:
            return self._load_cpu_llm(model_path)
        else:
            return self._load_npu_llm(model_path)

    def ask(self, question, max_tokens=128):
        """Ask the LLM a question. Returns text response."""
        if self._llm_mode == "cpu":
            return self._ask_cpu(question, max_tokens)
        elif self._llm_mode == "npu":
            return self._ask_npu(question, max_tokens)
        else:
            raise RuntimeError("[LLM] Not loaded. Call load_llm() first.")

    def ask_with_vision(self, image_bgr, question="What objects do you see in this image?"):
        """
        Full pipeline: YOLO detection + LLM interpretation.
        1. Detect objects in image
        2. Format results as context
        3. Ask LLM to interpret
        """
        objects = self.detect(image_bgr, conf_thresh=0.3)
        if objects:
            obj_list = ", ".join(
                "{} ({:.0%})".format(o["label"], o["confidence"])
                for o in sorted(objects, key=lambda x: x["confidence"], reverse=True)[:10]
            )
            prompt = "I detected these objects: {}. {}".format(obj_list, question)
        else:
            prompt = question

        return self.ask(prompt)

    # ================================================================
    #  Cleanup
    # ================================================================
    def close(self):
        if self._rknn:
            self._rknn.release()
            self._rknn = None
        if self._llm_mode == "npu" and self._llm_lib and self._llm_handle:
            self._llm_lib.rkllm_destroy(self._llm_handle)
            self._llm_handle = None
        if self._llm_mode == "cpu" and self._cpu_llm:
            self._cpu_llm.unload()
            self._cpu_llm = None

    # ================================================================
    #  Internal: NPU LLM
    # ================================================================
    def _load_npu_llm(self, model_path):
        path = model_path or self.NPU_LLM_MODEL
        if not os.path.isfile(path):
            print("[LLM-NPU] Model not found:", path)
            return False

        self._llm_lib = ctypes.CDLL("/usr/local/lib/librkllmrt.so")
        self._llm_lib.rkllm_createDefaultParam.restype = _P
        self._llm_lib.rkllm_init.argtypes = [POINTER(c_void_p), POINTER(_P), _CALLBACK]
        self._llm_lib.rkllm_init.restype = c_int32
        self._llm_lib.rkllm_run.argtypes = [c_void_p, POINTER(_INP), POINTER(_IP), c_void_p]
        self._llm_lib.rkllm_run.restype = c_int32
        self._llm_lib.rkllm_destroy.argtypes = [c_void_p]
        self._llm_lib.rkllm_destroy.restype = c_int32

        # Callback (module-level ref kept to avoid GC)
        self._llm_text_buf = []
        self._llm_ttft = 0.0
        self._llm_first = True
        def cb(r, u, s):
            if s >= 2: return
            now = time.time()
            if self._llm_first: self._llm_ttft = now; self._llm_first = False
            if r and r.contents.text:
                self._llm_text_buf.append(r.contents.text)
        self._llm_cb_fn = _CALLBACK(cb)

        param = self._llm_lib.rkllm_createDefaultParam()
        param.model_path = path.encode()
        param.max_context_len = 1024; param.max_new_tokens = 256
        param.top_k = 1; param.top_p = 0.9; param.temperature = 0.7
        param.repeat_penalty = 1.1; param.skip_special_token = True
        h = c_void_p()

        t0 = time.time()
        ret = self._llm_lib.rkllm_init(byref(h), byref(param), self._llm_cb_fn)
        if ret != 0:
            print("[LLM-NPU] Init failed:", ret)
            return False
        self._llm_handle = h
        self._llm_mode = "npu"
        print("[LLM-NPU] Loaded: {} ({:.0f}s)".format(os.path.basename(path), time.time()-t0))
        return True

    def _ask_npu(self, question, max_tokens):
        nl = chr(10)
        prompt = nl.join(["<|im_start|>user", question, "<|im_end|>", "<|im_start|>assistant", ""])

        self._llm_text_buf = []
        self._llm_first = True
        st = time.time()

        inp = _INP(); inp.input_type = 0; inp.prompt_input = prompt.encode()
        iparam = _IP(); iparam.mode = 0; iparam.lora_params = None; iparam.prompt_cache_params = None
        ret = self._llm_lib.rkllm_run(self._llm_handle, byref(inp), byref(iparam), None)
        total = time.time() - st

        raw = b"".join(self._llm_text_buf)
        text = raw.decode("utf-8", errors="replace").strip()
        ttft = self._llm_ttft - st if not self._llm_first else total
        return text

    # ================================================================
    #  Internal: CPU LLM
    # ================================================================
    def _load_cpu_llm(self, model_path):
        path = model_path or self.CPU_LLM_MODEL
        if not os.path.isfile(path):
            print("[LLM-CPU] Model not found:", path)
            return False
        try:
            from llama_cpp import Llama
            self._cpu_llm = Llama(model_path=path, n_ctx=2048, n_threads=4, verbose=False)
            self._llm_mode = "cpu"
            print("[LLM-CPU] Loaded:", os.path.basename(path))
            return True
        except Exception as e:
            print("[LLM-CPU] Error:", e)
            return False

    def _ask_cpu(self, question, max_tokens):
        im_s, im_e = "<|im_start|>", "<|im_end|>"
        nl = chr(10)
        prompt = nl.join([im_s + "user", question + im_e, im_s + "assistant", ""])

        result = self._cpu_llm.create_completion(
            prompt, max_tokens=max_tokens, temperature=0.7, stop=[im_e, im_s]
        )
        return result["choices"][0]["text"].strip()

    # ================================================================
    #  Internal: YOLOv5 Post-Processing
    # ================================================================
    @staticmethod
    def _yolov5_postprocess(outputs, orig_w, orig_h, conf_thresh, iou_thresh):
        strides = [8, 16, 32]
        all_boxes, all_scores, all_classes = [], [], []

        for i, output in enumerate(outputs):
            out = output[0]
            na, h, w = out.shape[0], out.shape[2], out.shape[3]
            nc = out.shape[1] - 5
            stride = strides[i]
            out = out.reshape(na, 5+nc, h*w).transpose(0,2,1).reshape(-1, 5+nc)

            grid_y, grid_x = np.mgrid[0:h, 0:w]
            grid_y = np.tile(grid_y.reshape(-1), 3)
            grid_x = np.tile(grid_x.reshape(-1), 3)

            sig = lambda x: 1/(1+np.exp(-x))
            out[:,0] = sig(out[:,0])*2 - 0.5 + grid_x
            out[:,1] = sig(out[:,1])*2 - 0.5 + grid_y
            out[:,2] = (sig(out[:,2])*2)**2
            out[:,3] = (sig(out[:,3])*2)**2
            out[:,0] *= stride; out[:,1] *= stride
            out[:,2] *= stride*4; out[:,3] *= stride*4

            obj = sig(out[:,4])
            cls_scores = sig(out[:,5:])
            scores = obj * cls_scores.max(axis=1)
            classes = cls_scores.argmax(axis=1)
            mask = scores > conf_thresh
            if not mask.any(): continue

            boxes = np.zeros((mask.sum(), 4))
            boxes[:,0] = out[mask,0] - out[mask,2]/2
            boxes[:,1] = out[mask,1] - out[mask,3]/2
            boxes[:,2] = out[mask,0] + out[mask,2]/2
            boxes[:,3] = out[mask,1] + out[mask,3]/2
            all_boxes.append(boxes)
            all_scores.append(scores[mask])
            all_classes.append(classes[mask])

        if not all_boxes: return []
        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        classes = np.concatenate(all_classes)

        # NMS
        x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
        areas = (x2-x1)*(y2-y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
            iou = w*h/(areas[i]+areas[order[1:]]-w*h)
            order = order[np.where(iou <= iou_thresh)[0] + 1]

        results = []
        for i in keep:
            x1 = int(boxes[i,0] * orig_w / 640)
            y1 = int(boxes[i,1] * orig_h / 640)
            x2 = int(boxes[i,2] * orig_w / 640)
            y2 = int(boxes[i,3] * orig_h / 640)
            cls_id = int(classes[i])
            label = ELFAI.COCO_NAMES[cls_id] if cls_id < len(ELFAI.COCO_NAMES) else str(cls_id)
            results.append({
                "label": label,
                "confidence": float(scores[i]),
                "box": [x1, y1, x2, y2],
            })
        return results


# ============================================================
#  Quick Test
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("ELF2 AI Engine — Quick Test")
    print("=" * 55)

    ai = ELFAI()

    # Test 1: YOLO detection
    print("\n[1] Loading YOLOv5s...")
    ai.load_yolo()

    import numpy as np
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    objects = ai.detect(img, conf_thresh=0.5)
    print("  Random image: {} detections (should be 0)".format(len(objects)))

    # Test 2: NPU LLM
    print("\n[2] Loading Qwen NPU LLM...")
    ai.load_llm(cpu=False)

    print("\n[3] Asking questions...")
    for q in ["What is 1+1?", "1+1等于几？"]:
        t0 = time.time()
        a = ai.ask(q, max_tokens=64)
        t = time.time() - t0
        print("  Q: {}".format(q))
        print("  A: {} ({:.1f}s)".format(a[:100], t))
        print()

    # Test 3: Vision + LLM pipeline
    print("[4] Vision+LLM pipeline...")
    a = ai.ask_with_vision(img, "What do you see?")
    print("  {}".format(a[:120]))

    ai.close()
    print("\nDone!")

#!/usr/bin/env python3
"""
NPU Vision Engine — RKNNLite wrapper for RK3588 NPU

Supports:
- Classification (ResNet18, MobileNetV2)
- Object Detection (YOLOX-S)

NPU memory is independent of system RAM.
Model files are under 15MB each.

Usage:
    vision = VisionNPU()
    vision.load_classifier("models/resnet18_for_rk3588.rknn")
    top5 = vision.classify(frame_bgr)
    vision.free()
"""
import cv2
import numpy as np
from rknnlite.api import RKNNLite
from typing import Optional, List, Tuple, Dict


class VisionNPU:
    def __init__(self):
        self._rknn: Optional[RKNNLite] = None
        self._loaded = False
        self._mode = ""           # "classify" or "detect"
        self._input_size = (0, 0)  # (H, W)
        self._input_fmt = ""       # NHWC or NCHW
        self._class_labels: Optional[List[str]] = None

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------
    def load_classifier(self, model_path: str,
                        labels: Optional[List[str]] = None) -> bool:
        return self._load(model_path, mode="classify", labels=labels)

    def load_detector(self, model_path: str) -> bool:
        return self._load(model_path, mode="detect")

    def free(self):
        if self._rknn:
            self._rknn.release()
            self._rknn = None
        self._loaded = False
        self._mode = ""

    def is_loaded(self) -> bool:
        return self._loaded

    def classify(self, image: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, str]]:
        if not self._loaded or self._mode != "classify":
            raise RuntimeError("Classifier not loaded. Call load_classifier() first.")
        inp = self._preprocess(image)
        outputs = self._rknn.inference(inputs=[inp])
        return self._post_classify(outputs[0], top_k)

    def detect(self, image: np.ndarray,
               conf_thresh: float = 0.25,
               nms_thresh: float = 0.45) -> List[Dict]:
        if not self._loaded or self._mode != "detect":
            raise RuntimeError("Detector not loaded. Call load_detector() first.")
        inp = self._preprocess(image)
        outputs = self._rknn.inference(inputs=[inp])
        return self._post_detect(outputs, image.shape[:2], conf_thresh, nms_thresh)

    # -------------------------------------------------------
    # Internal
    # -------------------------------------------------------
    # Known input dimensions for common RK3588 models
    _MODEL_INPUT_SIZES = {
        "resnet18":  (224, 224, "NCHW"),
        "mobilenet": (224, 224, "NCHW"),
        "yolox":     (640, 640, "NCHW"),
        "yolov5":    (640, 640, "NCHW"),
    }

    def _load(self, model_path: str, mode: str,
              labels: Optional[List[str]] = None) -> bool:
        import os
        basename = os.path.basename(model_path).lower()
        if not os.path.isfile(model_path):
            print(f"[VisionNPU] Model not found: {model_path}")
            return False

        self.free()
        rknn = RKNNLite()
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            print(f"[VisionNPU] load_rknn failed: {ret}")
            return False

        # Use core 0 on RK3588 (3 cores available)
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print(f"[VisionNPU] init_runtime failed: {ret}")
            rknn.release()
            return False

        self._rknn = rknn
        self._loaded = True
        self._mode = mode
        self._class_labels = labels

        # Detect input size from model name
        for key, (h, w, fmt) in self._MODEL_INPUT_SIZES.items():
            if key in basename:
                self._input_size = (h, w)
                self._input_fmt = fmt
                break
        else:
            self._input_size = (224, 224)
            self._input_fmt = "NCHW"

        mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"[VisionNPU] {mode} loaded ({mb:.1f}MB): {os.path.basename(model_path)}")
        return True

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w = self._input_size
        resized = cv2.resize(image, (w, h))
        if self._input_fmt == "NCHW":
            inp = np.transpose(resized, (2, 0, 1)).astype(np.float32)
        else:
            inp = resized.astype(np.float32)
        return np.expand_dims(inp, 0)

    def _post_classify(self, output: np.ndarray,
                       top_k: int) -> List[Tuple[int, float, str]]:
        probs = output.reshape(-1)
        probs = np.exp(probs - probs.max())       # stable softmax
        probs /= probs.sum()
        top_indices = np.argsort(probs)[::-1][:top_k]
        result = []
        for idx in top_indices:
            score = float(probs[idx])
            label = (self._class_labels[idx]
                     if self._class_labels and idx < len(self._class_labels)
                     else str(idx))
            result.append((idx, score, label))
        return result

    # -------------------------------------------------------
    # YOLOX-S post-processing
    # -------------------------------------------------------
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

    YOLOX_STRIDES = [8, 16, 32]

    def _post_detect(self, outputs: List[np.ndarray], orig_shape,
                     conf_thresh: float, nms_thresh: float) -> List[Dict]:
        all_boxes: List[Tuple[float, float, float, float]] = []
        all_scores: List[float] = []
        all_classes: List[int] = []

        in_h, in_w = self._input_size
        orig_h, orig_w = orig_shape

        # Each output is a multi-scale detection head
        num_heads = len(outputs)  # typically 3 for YOLOX-S
        strides = self.YOLOX_STRIDES[-num_heads:] if num_heads <= 3 else [8, 16, 32, 64][-num_heads:]

        for head_idx, output in enumerate(outputs):
            stride = strides[head_idx] if head_idx < len(strides) else 8 * (2 ** head_idx)
            grid_h = in_h // stride
            grid_w = in_w // stride
            num_classes = output.shape[-1] - 5 if len(output.shape) == 3 else 80

            feat = output.reshape(grid_h, grid_w, -1)

            # Decode boxes
            for gy in range(grid_h):
                for gx in range(grid_w):
                    row = feat[gy, gx]
                    box_conf = float(row[4])
                    if box_conf < conf_thresh:
                        continue

                    max_cls_prob = 0.0
                    max_cls_idx = 0
                    for c in range(num_classes):
                        if 5 + c < len(row):
                            p = float(row[5 + c])
                            if p > max_cls_prob:
                                max_cls_prob = p
                                max_cls_idx = c

                    score = box_conf * max_cls_prob
                    if score < conf_thresh:
                        continue

                    # Decode bounding box
                    cx = (float(row[0]) + gx) * stride
                    cy = (float(row[1]) + gy) * stride
                    bw = float(row[2]) * stride
                    bh = float(row[3]) * stride

                    x1 = max(0, int(cx - bw / 2))
                    y1 = max(0, int(cy - bh / 2))
                    x2 = min(in_w, int(cx + bw / 2))
                    y2 = min(in_h, int(cy + bh / 2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    all_boxes.append((x1, y1, x2, y2))
                    all_scores.append(score)
                    all_classes.append(max_cls_idx)

        # Apply NMS
        kept = self._nms(all_boxes, all_scores, all_classes, nms_thresh)

        results: List[Dict] = []
        for i in kept:
            x1, y1, x2, y2 = all_boxes[i]
            # Rescale to original image coordinates
            x1 = int(x1 * orig_w / in_w)
            y1 = int(y1 * orig_h / in_h)
            x2 = int(x2 * orig_w / in_w)
            y2 = int(y2 * orig_h / in_h)

            cls_id = all_classes[i]
            label = (self.COCO_NAMES[cls_id]
                     if cls_id < len(self.COCO_NAMES)
                     else f"class_{cls_id}")

            results.append({
                "label": label,
                "class_id": cls_id,
                "confidence": round(float(all_scores[i]), 4),
                "box": (x1, y1, x2, y2),
            })

        return results

    # NMS (Non-Maximum Suppression)
    @staticmethod
    def _nms(boxes, scores, classes, iou_thresh):
        if not boxes:
            return []
        x1 = np.array([b[0] for b in boxes], dtype=np.float32)
        y1 = np.array([b[1] for b in boxes], dtype=np.float32)
        x2 = np.array([b[2] for b in boxes], dtype=np.float32)
        y2 = np.array([b[3] for b in boxes], dtype=np.float32)
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

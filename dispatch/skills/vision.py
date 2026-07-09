"""Vision Skill — camera capture + YOLOv5s detection + LLM interpretation."""
import ctypes, time, os, numpy as np
from ctypes import *
from .base import Skill

# YOLO post-process helpers
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


class VisionSkill(Skill):
    name = "vision"
    description = "Camera object detection - captures image, runs YOLOv5s on NPU, interprets results"
    requires_camera = True
    requires_eeg = False

    def __init__(self, camera_id=21, chat_skill=None):
        self._rknn = None
        self._camera_id = camera_id
        self._chat = chat_skill  # LLM for interpreting results
        self._cap = None

    def load(self) -> bool:
        from rknnlite.api import RKNNLite
        model = "/home/elf/Projects/models/yolov5s-640-640.rknn"
        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(model)
        if ret != 0:
            print("[Vision] load_rknn failed:", ret)
            return False
        ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print("[Vision] init_runtime failed:", ret)
            return False

        # Open camera
        import cv2
        self._cap = cv2.VideoCapture(self._camera_id)
        if not self._cap.isOpened():
            print("[Vision] Camera /dev/video{} not available".format(self._camera_id))
            # Don't fail - vision skill is still "loaded", just no camera
        else:
            print("[Vision] Camera /dev/video{} opened".format(self._camera_id))
        print("[Vision] YOLOv5s loaded on NPU")
        return True

    def is_ready(self) -> bool:
        return self._rknn is not None

    def execute(self, question: str, **kwargs) -> str:
        """Capture image, detect objects, ask LLM to interpret."""
        # 1. Capture
        if not self._cap or not self._cap.isOpened():
            return "[Vision] Camera not available. Please connect a USB camera."

        ret, frame = self._cap.read()
        if not ret:
            return "[Vision] Failed to capture image from camera."

        # 2. YOLO detection
        objects = self._detect(frame)
        print("[Vision] Detected {} objects".format(len(objects)))

        # 3. Build context
        if objects:
            obj_list = ", ".join(
                "{} ({:.0%})".format(o["label"], o["confidence"])
                for o in sorted(objects, key=lambda x: x["confidence"], reverse=True)[:10]
            )
            context = "I see these objects in the camera: {}. ".format(obj_list)
        else:
            context = "I looked at the camera but did not detect any recognizable objects. "

        prompt = context + question
        print("[Vision] Prompt:", prompt[:120])

        # 4. LLM interpretation
        if self._chat:
            return self._chat.execute(prompt)
        else:
            return prompt + "\n(No LLM loaded for interpretation)"

    def capture_only(self):
        """Capture and detect, return raw results (no LLM)."""
        if not self._cap or not self._cap.isOpened():
            return []
        ret, frame = self._cap.read()
        if not ret:
            return []
        return self._detect(frame)

    def cleanup(self):
        if self._rknn:
            self._rknn.release()
        if self._cap:
            self._cap.release()

    # ---- Internal ----
    def _detect(self, bgr_image):
        import cv2
        orig_h, orig_w = bgr_image.shape[:2]
        resized = cv2.resize(bgr_image, (640, 640))
        inp = np.transpose(resized, (2, 0, 1)).astype(np.float32)[None, :]
        outputs = self._rknn.inference(inputs=[inp])
        return self._postprocess(outputs, orig_w, orig_h)

    @staticmethod
    def _postprocess(outputs, orig_w, orig_h, conf_thresh=0.35, iou_thresh=0.45):
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
            label = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else str(cls_id)
            results.append({"label": label, "confidence": float(scores[i]), "box": [x1, y1, x2, y2]})
        return results

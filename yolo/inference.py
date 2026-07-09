#!/usr/bin/env python3
"""
YOLO 情绪识别模块
两阶段: 人脸检测 (yolov11n-face) → 情绪分类 (affectnet_best)
6类 → 3类映射: happy→positive, neutral→neutral, 其余→negative
"""
import os
import numpy as np
from ultralytics import YOLO

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

CLASS_6_TO_3 = {
    'happy':    'positive',
    'neutral':  'neutral',
    'anger':    'negative',
    'disgust':  'negative',
    'sad':      'negative',
    'surprise': 'negative',
}

THREE_CLASS_ORDER = ['positive', 'neutral', 'negative']


class YoloEmotionRecognizer:
    def __init__(self, face_model=None, emotion_model=None, imgsz_face=640,
                 imgsz_emotion=224, conf_face=0.25, conf_emotion=0.25,
                 device='cpu'):
        face_path = face_model or os.path.join(MODEL_DIR, 'yolov11n-face.pt')
        emo_path = emotion_model or os.path.join(MODEL_DIR, 'affectnet_best.pt')

        self.device = device
        self.imgsz_face = imgsz_face
        self.imgsz_emotion = imgsz_emotion
        self.conf_face = conf_face
        self.conf_emotion = conf_emotion

        self.face_model = YOLO(face_path)
        self.emo_model = YOLO(emo_path)
        self.emo_names = list(self.emo_model.names.values())

        self._warmed = False
        self._warmup()

    def _warmup(self):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.face_model(dummy, imgsz=self.imgsz_face, verbose=False,
                        device=self.device)
        self.emo_model(dummy, imgsz=self.imgsz_emotion, verbose=False,
                       device=self.device)
        self._warmed = True

    def detect_emotions(self, frame):
        """
        输入: BGR 图像 (numpy array, H×W×3)
        返回: [(bbox, [pos, neu, neg], conf), ...]  或  [](空列表) 若无人脸
            bbox: (x1, y1, x2, y2) 归一化 [0,1]
            [pos, neu, neg]: 三分类概率 (sum=1)
            conf: 平均置信度
        """
        face_results = self.face_model(frame, imgsz=self.imgsz_face,
                                       conf=self.conf_face, verbose=False,
                                       device=self.device)

        emotions = []
        for r in face_results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                face_crop = frame[y1:y2, x1:x2]

                emo_results = self.emo_model(face_crop, imgsz=self.imgsz_emotion,
                                             conf=self.conf_emotion, verbose=False,
                                             device=self.device)

                probs = emo_results[0].probs
                if probs is None:
                    continue

                cls_id = int(probs.top1)
                conf = float(probs.top1conf)
                label = self.emo_names[cls_id]

                prob = self._six_to_three_prob(label, conf)

                h, w = frame.shape[:2]
                bbox_norm = (x1/w, y1/h, x2/w, y2/h)
                emotions.append((bbox_norm, prob.tolist(), conf))

        return emotions

    def _six_to_three_prob(self, label, confidence):
        """
        将 6 分类标签转换为 3 分类概率
        使用 label smoothing: 给预测类别赋予 confidence,
        剩余 (1-confidence) 均匀分配给同 3-category 的其他类
        """
        prob = np.zeros(3, dtype=np.float32)
        three_class = CLASS_6_TO_3.get(label, 'neutral')
        idx = THREE_CLASS_ORDER.index(three_class)
        prob[idx] = confidence

        residual = 1.0 - confidence
        for i in range(3):
            if i != idx:
                prob[i] = residual / 2.0

        return prob

    def get_primary_emotion(self, frame):
        """
        便捷方法: 返回主人脸的三分类概率, 无人脸返回 None
        """
        results = self.detect_emotions(frame)
        if not results:
            return None

        results.sort(key=lambda x: x[1][0] + x[1][2], reverse=True)
        return results[0][1]

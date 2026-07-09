#!/usr/bin/env python3
"""
YOLO 情绪识别测试脚本
用法: python test_yolo.py [--camera 0] [--image path/to/img.jpg]
"""
import sys, os, time, argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import YoloEmotionRecognizer
from fusion import fuse

CLASS_NAMES_CN = ['积极', '中性', '消极']
COLORS = [(0, 255, 0), (200, 200, 200), (0, 0, 255)]


def draw_results(frame, emotions):
    h, w = frame.shape[:2]
    for bbox_norm, probs, conf in emotions:
        x1, y1, x2, y2 = [int(v * s) for v, s in zip(bbox_norm, [w, h, w, h])]
        best_idx = int(np.argmax(probs))
        label = CLASS_NAMES_CN[best_idx]
        color = COLORS[best_idx]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {probs[best_idx]:.2f}"
        cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def run_image(path):
    recognizer = YoloEmotionRecognizer()
    frame = cv2.imread(path)
    if frame is None:
        print(f"无法读取: {path}")
        return

    t0 = time.time()
    results = recognizer.detect_emotions(frame)
    elapsed = (time.time() - t0) * 1000

    print(f"推理耗时: {elapsed:.1f}ms, 检测到 {len(results)} 张人脸")
    for i, (bbox, probs, conf) in enumerate(results):
        print(f"  人脸{i+1}: bbox={[f'{v:.2f}' for v in bbox]}, "
              f"积极={probs[0]:.3f} 中性={probs[1]:.3f} 消极={probs[2]:.3f}, "
              f"置信度={conf:.2f}")

    frame = draw_results(frame, results)
    out_path = '/tmp/yolo_emotion_result.jpg'
    cv2.imwrite(out_path, frame)
    print(f"结果已保存: {out_path}")


def run_camera(cam_id=0):
    recognizer = YoloEmotionRecognizer()
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 {cam_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("按 Q 退出 | 按 S 截图保存")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        results = recognizer.detect_emotions(frame)
        fps = 1.0 / max(time.time() - t0, 0.001)

        frame = draw_results(frame, results)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('YOLO Emotion Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('/tmp/yolo_emotion_screenshot.jpg', frame)
            print("截图已保存")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO 情绪识别测试')
    parser.add_argument('--camera', type=int, default=None, help='摄像头编号')
    parser.add_argument('--image', type=str, default=None, help='图片路径')
    args = parser.parse_args()

    if args.image:
        run_image(args.image)
    else:
        cam = args.camera if args.camera is not None else 21
        run_camera(cam)

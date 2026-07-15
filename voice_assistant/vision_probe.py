"""One-shot YOLO probe used only for explicit visual voice requests."""

from __future__ import annotations

import json
import os
import sys
import time


def main() -> int:
    camera_id = int(os.environ.get("VOICE_CAMERA_ID", "11"))
    skill = None
    try:
        # Do not allocate the YOLO runtime unless the camera has a real frame.
        import cv2
        camera = cv2.VideoCapture(camera_id)
        try:
            time.sleep(0.15)
            ok, frame = camera.read()
        finally:
            camera.release()
        if not ok or frame is None:
            raise RuntimeError(f"camera /dev/video{camera_id} did not provide a frame")

        from dispatch.skills.vision import VisionSkill

        skill = VisionSkill(camera_id=camera_id)
        if not skill.load():
            raise RuntimeError("YOLO load failed")
        objects = skill._detect(frame)
        result = [
            {"label": item["label"], "confidence": round(float(item["confidence"]), 2)}
            for item in sorted(objects, key=lambda item: item["confidence"], reverse=True)[:8]
        ]
        print(json.dumps({"ok": True, "objects": result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    finally:
        if skill is not None:
            skill.cleanup()


if __name__ == "__main__":
    sys.exit(main())

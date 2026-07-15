#!/usr/bin/env python3
"""One-shot pull worker for server-to-ELF2 knowledge and RKNN updates."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SERVER = os.environ.get("BOARD_SYNC_SERVER", "http://100.99.254.18:8090")
SOCKS = os.environ.get("BOARD_SYNC_SOCKS", "127.0.0.1:1055")
BOARD_ID = os.environ.get("BOARD_SYNC_ID", "elf2-board")
APP_DIR = Path("/home/elf/Projects/voice_assistant")
CACHE = Path("/home/elf/.cache/board-sync")
CUSTOM_KNOWLEDGE = APP_DIR / "custom_knowledge"
YOLO_MODEL = Path("/home/elf/Projects/models/yolov5s-640-640.rknn")
YOLO_LABELS = Path("/home/elf/Projects/models/yolo_labels.json")


def curl(path: str, output: Path | None = None, payload: dict | None = None) -> str:
    command = ["curl", "--silent", "--show-error", "--fail", "--max-time", "45", "--socks5-hostname", SOCKS]
    if payload is not None:
        command.extend(["-X", "POST", "-H", "Content-Type: application/json", "--data", json.dumps(payload)])
    if output is not None:
        command.extend(["--output", str(output)])
    command.append(f"{SERVER}{path}")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or "network request failed").strip())
    return result.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acknowledge(item_id: str, status: str, detail: str = "") -> None:
    curl(f"/api/board/queue/{item_id}/ack", payload={"status": status, "detail": detail[:500]})


def install_knowledge(source: Path, item: dict) -> None:
    document = json.loads(source.read_text(encoding="utf-8"))
    if not all(isinstance(document.get(key), value_type) for key, value_type in (("id", str), ("title", str), ("content", str), ("keywords", list))):
        raise RuntimeError("invalid knowledge document")
    CUSTOM_KNOWLEDGE.mkdir(parents=True, exist_ok=True)
    target = CUSTOM_KNOWLEDGE / f"{item['id']}.json"
    os.replace(source, target)
    subprocess.run([sys.executable, str(APP_DIR / "sync_custom_knowledge.py")], check=True, timeout=20)
    if item.get("restart_service"):
        subprocess.run(
            ["systemctl", "--user", "restart", "voice-assistant.service"],
            check=True,
            timeout=30,
            env={**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"},
        )


def install_yolov5_rknn(source: Path, item: dict) -> None:
    subprocess.run([sys.executable, str(APP_DIR / "validate_rknn.py"), str(source)], check=True, timeout=35)
    YOLO_MODEL.parent.mkdir(parents=True, exist_ok=True)
    labels = item.get("class_names")
    if labels:
        if not isinstance(labels, list) or not all(isinstance(label, str) and label for label in labels):
            raise RuntimeError("invalid YOLO class metadata")
        temporary = YOLO_LABELS.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, YOLO_LABELS)
    backup = YOLO_MODEL.with_suffix(YOLO_MODEL.suffix + ".previous")
    if YOLO_MODEL.exists():
        os.replace(YOLO_MODEL, backup)
    os.replace(source, YOLO_MODEL)


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        response = json.loads(curl(f"/api/board/queue?board_id={BOARD_ID}"))
        item = response.get("item")
        if not item:
            return 0
        item_id = item["id"]
        with tempfile.NamedTemporaryFile(prefix=f"{item_id}-", suffix=".part", dir=CACHE, delete=False) as handle:
            source = Path(handle.name)
        try:
            curl(item["download_url"], source)
            if sha256(source) != item["sha256"]:
                raise RuntimeError("sha256 verification failed")
            if item["kind"] == "knowledge":
                install_knowledge(source, item)
            elif item["kind"] == "yolov5-rknn":
                install_yolov5_rknn(source, item)
            else:
                raise RuntimeError(f"unsupported update type: {item['kind']}")
            acknowledge(item_id, "delivered", "installed and verified")
        except Exception as error:
            source.unlink(missing_ok=True)
            acknowledge(item_id, "failed", str(error))
            raise
    except Exception as error:
        print(f"[board-sync] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

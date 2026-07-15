#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if PYTHONPATH="$ROOT/.tooling/yolo-rknn-site" python3 "$ROOT/.tooling/yolov5/train.py" --help >"$ROOT/.tooling/yolov5-env-error.txt" 2>&1; then
  printf 'yolov5-v7-training-entrypoint-ok\n' > "$ROOT/.tooling/yolov5-env-ok.txt"
else
  rm -f "$ROOT/.tooling/yolov5-env-ok.txt"
fi

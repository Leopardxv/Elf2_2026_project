#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET="$ROOT/.tooling/yolov5"
ARCHIVE="$ROOT/.tooling/yolov5-v7.0.tar.gz"

rm -rf "$TARGET" "$ARCHIVE"
curl --proxy http://127.0.0.1:7890 --connect-timeout 10 --retry 2 -fL \
  https://codeload.github.com/ultralytics/yolov5/tar.gz/refs/tags/v7.0 -o "$ARCHIVE"
tar -xzf "$ARCHIVE" -C "$ROOT/.tooling"
mv "$ROOT/.tooling/yolov5-7.0" "$TARGET"
rm "$ARCHIVE"
test -f "$TARGET/train.py"
test -f "$TARGET/export.py"
echo "YOLOv5 v7.0 ready: $TARGET"

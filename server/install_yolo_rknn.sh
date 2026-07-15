#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SITE="$ROOT/.tooling/yolo-rknn-site"
PROXY="http://127.0.0.1:7890"
WHEEL="$ROOT/.tooling/rknn-wheels/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

export HTTPS_PROXY="$PROXY"
export HTTP_PROXY="$PROXY"

python3 -m pip install --upgrade --target "$SITE" --no-deps \
  "numpy==1.26.4" \
  "protobuf==4.25.4" \
  "ruamel.yaml>=0.17.21" \
  "fast-histogram>=0.11" \
  "onnx==1.16.1" \
  "onnxruntime==1.18.1" \
  "opencv-python==4.10.0.84" \
  "ultralytics==8.3.0" \
  "onnxsim==0.4.36" \
  "$WHEEL"

python3 -m pip install --upgrade --target "$SITE" \
  "numpy==1.26.4" "pandas==2.2.2" "seaborn==0.13.2" "ipython" "gitpython"

PYTHONPATH="$SITE" python3 - <<'PY'
import numpy
import onnx
import rknn
import torch
import torchvision
import ultralytics

assert torch.cuda.is_available(), "CUDA is unavailable"
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.get_device_name(0))
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics.__version__)
print("rknn-toolkit2 import OK")
PY

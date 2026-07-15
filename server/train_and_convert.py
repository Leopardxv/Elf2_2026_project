#!/usr/bin/env python3
"""GPU YOLOv5 training followed by an RK3588 INT8 RKNN export."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def image_files(dataset_root: Path, data_file: Path, limit: int = 200) -> list[Path]:
    import yaml

    config = yaml.safe_load(data_file.read_text(encoding="utf-8")) or {}
    dataset_path = Path(config.get("path") or dataset_root)
    if not dataset_path.is_absolute():
        dataset_path = (data_file.parent / dataset_path).resolve()
    train = config.get("train")
    values = train if isinstance(train, list) else [train]
    results = []
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for value in values:
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = dataset_path / path
        if path.is_dir():
            results.extend(candidate for candidate in sorted(path.rglob("*")) if candidate.suffix.lower() in suffixes)
        elif path.suffix.lower() == ".txt" and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                candidate = Path(line.strip())
                if candidate and not candidate.is_absolute():
                    candidate = dataset_root / candidate
                if candidate.is_file() and candidate.suffix.lower() in suffixes:
                    results.append(candidate)
        if len(results) >= limit:
            break
    return results[:limit]


def class_names(data_file: Path) -> list[str]:
    import yaml

    names = (yaml.safe_load(data_file.read_text(encoding="utf-8")) or {}).get("names", [])
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=lambda value: int(value))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise RuntimeError("data.yaml must define class names for board deployment")


def training_data_file(dataset_root: Path, output_root: Path) -> Path:
    """Resolve dataset-relative paths before invoking YOLOv5 from its own repo."""
    import yaml

    source = dataset_root / "data.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    config["path"] = str(dataset_root.resolve())
    prepared = output_root / "training-data.yaml"
    prepared.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return prepared


def ensure_yolov5_font():
    """Avoid YOLOv5's optional network fetch for Arial.ttf in isolated jobs."""
    target = Path.home() / ".config" / "Ultralytics" / "Arial.ttf"
    if target.is_file():
        return
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ):
        if candidate.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            print(f"[Setup] Local font cache: {target}")
            return
    raise RuntimeError("YOLOv5 needs a local TrueType font, but none was found on the server")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="yolov5s.pt")
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--imgsz", required=True, type=int)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    source_data_file = args.dataset / "data.yaml"
    if not source_data_file.is_file():
        raise FileNotFoundError(f"missing dataset configuration: {source_data_file}")
    args.output.mkdir(parents=True, exist_ok=True)
    data_file = training_data_file(args.dataset, args.output)
    ensure_yolov5_font()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the isolated training environment")
    yolov5_dir = Path(os.environ.get("YOLOV5_DIR", ""))
    if not (yolov5_dir / "train.py").is_file() or not (yolov5_dir / "export.py").is_file():
        raise RuntimeError("official YOLOv5 v7.0 tooling is unavailable")
    best = args.output / "run" / "weights" / "best.pt"
    if not args.skip_training:
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print("[Train] cache=False, workers=8, device=0")
        subprocess.run([
            sys.executable, str(yolov5_dir / "train.py"), "--data", str(data_file), "--weights", args.model,
            "--epochs", str(args.epochs), "--img", str(args.imgsz), "--project", str(args.output), "--name", "run",
            "--exist-ok", "--device", "0", "--workers", "8", "--batch-size", "16",
        ], cwd=yolov5_dir, check=True)
    else:
        print("[Train] Reusing existing best.pt; skipping completed training stage.")
    if not best.is_file():
        raise RuntimeError("training completed without best.pt")
    onnx_model = best.with_suffix(".onnx")
    if not onnx_model.is_file():
        print("[Export] ONNX opset=12, static input")
        subprocess.run([
            sys.executable, str(yolov5_dir / "export.py"), "--weights", str(best), "--imgsz", str(args.imgsz),
            "--include", "onnx", "--opset", "12", "--simplify",
        ], cwd=yolov5_dir, check=True)
    if not onnx_model.is_file():
        raise RuntimeError("ONNX export did not produce a file")

    calibration_images = image_files(args.dataset, data_file)
    if not calibration_images:
        raise RuntimeError("no training images available for INT8 calibration")
    calibration = args.output / "calibration.txt"
    calibration.write_text("\n".join(str(path) for path in calibration_images) + "\n", encoding="utf-8")
    (args.output / "classes.json").write_text(json.dumps(class_names(source_data_file), ensure_ascii=False), encoding="utf-8")
    print(f"[RKNN] INT8 calibration images: {len(calibration_images)}")

    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        if rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588") != 0:
            raise RuntimeError("RKNN config failed")
        if rknn.load_onnx(model=str(onnx_model)) != 0:
            raise RuntimeError("RKNN ONNX load failed")
        if rknn.build(do_quantization=True, dataset=str(calibration)) != 0:
            raise RuntimeError("RKNN INT8 build failed")
        artifact = args.output / "best.rknn"
        if rknn.export_rknn(str(artifact)) != 0:
            raise RuntimeError("RKNN export failed")
    finally:
        rknn.release()

    print(f"[Success] {artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

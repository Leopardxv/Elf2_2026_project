#!/usr/bin/env python3
"""Static SPA host plus the small training and board-distribution API."""

import cgi
import hashlib
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PORT = int(os.environ.get("PORT", "8090"))
ROOT = Path(__file__).resolve().parent
DIST = ROOT / "serever" / "dist"
RUNTIME = ROOT / "runtime"
KNOWLEDGE = RUNTIME / "knowledge"
DATASETS = RUNTIME / "datasets"
JOBS = RUNTIME / "jobs"
ARTIFACTS = RUNTIME / "artifacts"
QUEUE = RUNTIME / "queue.json"
CONFIG = RUNTIME / "board.json"
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
MAX_DATASET_FILES = 20_000
MAX_EXTRACTED_DATASET_BYTES = 2 * 1024 * 1024 * 1024
GPU_READY_MAX_UTILIZATION = 20
GPU_READY_MAX_MEMORY_MIB = 2048
GPU_WAIT_TIMEOUT_SECONDS = 2 * 60 * 60
GPU_WAIT_INTERVAL_SECONDS = 30

for directory in (RUNTIME, KNOWLEDGE, DATASETS, JOBS, ARTIFACTS):
    directory.mkdir(parents=True, exist_ok=True)


class ApiError(Exception):
    def __init__(self, message, status=HTTPStatus.BAD_REQUEST):
        self.message = message
        self.status = status


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def safe_name(name, fallback="artifact"):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return (cleaned or fallback)[:120]


def safe_relative(name):
    normalized = name.replace("\\", "/").lstrip("/")
    path = Path(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ApiError("数据集包含不安全的路径")
    return path


def unpack_dataset_zip(item, target):
    """Safely unpack a single archive uploaded through the training page."""
    try:
        archive = zipfile.ZipFile(item.file)
    except zipfile.BadZipFile:
        raise ApiError("上传的 ZIP 文件损坏或格式无效")
    with archive:
        members = [member for member in archive.infolist() if not member.is_dir() and not member.filename.startswith("__MACOSX/")]
        if not members:
            raise ApiError("ZIP 中没有可用文件")
        if len(members) > MAX_DATASET_FILES:
            raise ApiError("ZIP 内文件数量超过限制")
        if sum(member.file_size for member in members) > MAX_EXTRACTED_DATASET_BYTES:
            raise ApiError("ZIP 解压后的数据量超过限制")
        for member in members:
            relative = safe_relative(member.filename)
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, output.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    return len(members)


def locate_dataset_yaml(target):
    candidates = sorted(path for path in target.rglob("data.yaml") if path.is_file())
    if not candidates:
        raise ApiError("数据集缺少 data.yaml，无法开始训练。")
    if len(candidates) > 1:
        raise ApiError("数据集包含多个 data.yaml，无法确定训练根目录。")
    return candidates[0]


def available_training_gpu():
    """Return an idle physical GPU without changing any global CUDA setting."""
    forced_gpu = os.environ.get("YOLO_TRAINING_GPU")
    if forced_gpu:
        return forced_gpu, f"使用 YOLO_TRAINING_GPU={forced_gpu} 指定的显卡。"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        candidates = []
        for line in output.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) != 3:
                continue
            index, memory_used, utilization = (int(value) for value in values)
            candidates.append((utilization, memory_used, index))
        if not candidates:
            return None, "未能读取 NVIDIA GPU 状态。"
        utilization, memory_used, index = min(candidates)
        detail = f"GPU {index}: 利用率 {utilization}%，显存 {memory_used} MiB。"
        if utilization <= GPU_READY_MAX_UTILIZATION and memory_used <= GPU_READY_MAX_MEMORY_MIB:
            return str(index), f"{detail} 可启动训练。"
        return None, f"{detail} 正在忙，继续等待空闲 GPU。"
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return None, f"无法检查 GPU 状态：{error}"


def validate_board_host(host):
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,251}[A-Za-z0-9]", host or ""):
            raise ApiError("板子地址必须是 IP 地址或合法主机名")
        return host


def board_config(payload=None):
    config = read_json(CONFIG, {})
    payload = payload or {}
    if payload.get("board_host"):
        config["host"] = validate_board_host(payload["board_host"].strip())
    config.setdefault("user", os.environ.get("BOARD_SSH_USER", "elf"))
    config.setdefault("port", int(os.environ.get("BOARD_SSH_PORT", "22")))
    config.setdefault("knowledge_path", os.environ.get("BOARD_KNOWLEDGE_PATH", "/home/elf/Projects/voice_assistant/knowledge_base"))
    config.setdefault("yolo_path", os.environ.get("BOARD_YOLO_PATH", "/home/elf/Projects/models/yolo"))
    if config.get("host"):
        write_json(CONFIG, config)
    return config


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def title_keywords(title):
    """Generate compact Chinese title n-grams for the board's lightweight matcher."""
    chunks = re.findall(r"[\u4e00-\u9fff]{2,}", title)
    keywords = set()
    for chunk in chunks:
        for width in range(2, min(5, len(chunk) + 1)):
            keywords.update(chunk[index:index + width] for index in range(len(chunk) - width + 1))
    keywords.update(re.findall(r"[A-Za-z0-9_-]{2,}", title.lower()))
    return sorted(keywords)[:80]


def enqueue_artifact(kind, source, filename, board_id="elf2-board", **extra):
    if not source.is_file():
        raise ApiError("待发布文件不存在", HTTPStatus.NOT_FOUND)
    artifact_id = str(uuid.uuid4())
    destination = ARTIFACTS / f"{artifact_id}{source.suffix.lower()}"
    shutil.copy2(source, destination)
    record = {
        "id": artifact_id,
        "kind": kind,
        "board_id": board_id,
        "filename": safe_name(filename),
        "artifact": str(destination.relative_to(ROOT)),
        "sha256": sha256_file(destination),
        "size": destination.stat().st_size,
        "status": "queued",
        "created_at": time.time(),
        **extra,
    }
    queue = read_json(QUEUE, [])
    queue.append(record)
    write_json(QUEUE, queue)
    return record


def queue_item(item_id):
    return next((item for item in read_json(QUEUE, []) if item.get("id") == item_id), None)


def update_queue(item_id, **changes):
    queue = read_json(QUEUE, [])
    for item in queue:
        if item.get("id") == item_id:
            item.update(changes)
            item["updated_at"] = time.time()
            write_json(QUEUE, queue)
            return item
    return None


def job_path(job_id):
    if not re.fullmatch(r"[a-f0-9-]{36}", job_id):
        raise ApiError("任务编号无效", HTTPStatus.NOT_FOUND)
    return JOBS / f"{job_id}.json"


def update_job(job_id, **changes):
    path = job_path(job_id)
    job = read_json(path, None)
    if not job:
        return
    job.update(changes)
    job["updated_at"] = time.time()
    write_json(path, job)


def training_progress(line, epochs):
    match = re.match(r"\s*(\d+)/(\d+)\s+", line)
    if not match:
        return None
    current = min(int(match.group(1)) + 1, epochs)
    return {
        "stage": "training",
        "current_epoch": current,
        "total_epochs": epochs,
        "percent": round(current / epochs * 100, 1),
    }


def run_training(job_id, dataset_dir, config):
    log = ["[Queue] 等待低负载 GPU，避免影响服务器上已有任务。"]
    deadline = time.time() + GPU_WAIT_TIMEOUT_SECONDS
    gpu_index = None
    while time.time() < deadline:
        gpu_index, detail = available_training_gpu()
        log.append(f"[GPU Scheduler] {detail}")
        if gpu_index is not None:
            break
        update_job(job_id, status="queued", logs=log[-160:], progress={"stage": "waiting_gpu", "current_epoch": 0, "total_epochs": config["epochs"], "percent": 0})
        time.sleep(GPU_WAIT_INTERVAL_SECONDS)
    if gpu_index is None:
        update_job(job_id, status="failed", logs=log[-160:], error="等待空闲 GPU 超时，未启动训练。")
        return
    log.append("[System] 正在检查隔离的 GPU 训练与 RKNN 转换环境。")
    started_at = time.time()
    update_job(job_id, status="running", logs=log[-160:], started_at=started_at, progress={"stage": "training", "current_epoch": 0, "total_epochs": config["epochs"], "percent": 0})
    tooling_site = ROOT / ".tooling" / "yolo-rknn-site"
    runner = ROOT / "train_and_convert.py"
    if not tooling_site.is_dir() or not runner.is_file():
        update_job(job_id, status="failed", logs=log + ["[Error] GPU/RKNN 隔离环境尚未完成安装。"], error="训练工具链尚未就绪。")
        return
    output_root = JOBS / job_id
    output_root.mkdir(exist_ok=True)
    command = [
        sys.executable, str(runner), "--dataset", str(dataset_dir), "--output", str(output_root),
        "--model", config["model"], "--epochs", str(config["epochs"]), "--imgsz", str(config["image_size"]),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{tooling_site}{os.pathsep}{environment.get('PYTHONPATH', '')}"
    environment["CUDA_VISIBLE_DEVICES"] = gpu_index
    environment.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
    environment.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
    environment["YOLOv5_AUTOINSTALL"] = "false"
    environment.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".tooling" / "ultralytics"))
    environment["YOLOV5_DIR"] = str(ROOT / ".tooling" / "yolov5")
    environment["HOME"] = str(ROOT / ".tooling" / "training-home")
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            text = line.rstrip()
            log.append(text)
            progress = training_progress(text, config["epochs"])
            if text.startswith("[Export]"):
                progress = {"stage": "exporting", "current_epoch": config["epochs"], "total_epochs": config["epochs"], "percent": 100}
            elif text.startswith("[RKNN]"):
                progress = {"stage": "quantizing", "current_epoch": config["epochs"], "total_epochs": config["epochs"], "percent": 100}
            changes = {"logs": log[-160:]}
            if progress:
                changes["progress"] = progress
            update_job(job_id, **changes)
        if process.wait() != 0:
            update_job(job_id, status="failed", logs=log[-160:], error="YOLO 训练进程异常退出。")
            return
        artifact = output_root / "best.rknn"
        if not artifact.exists() or artifact.stat().st_size < 1024:
            update_job(job_id, status="failed", logs=log[-160:], error="训练或 RKNN 转换完成，但未找到有效的 best.rknn。")
            return
        best = output_root / "run" / "weights" / "best.pt"
        update_job(
            job_id, status="completed", logs=log[-160:] + ["[Success] 已生成并量化为 RK3588 可用的 best.rknn。"],
            artifact_pt=str(best.relative_to(ROOT)) if best.exists() else None,
            artifact_rknn=str(artifact.relative_to(ROOT)),
            artifact_labels=str((output_root / "classes.json").relative_to(ROOT)),
            progress={"stage": "completed", "current_epoch": config["epochs"], "total_epochs": config["epochs"], "percent": 100},
        )
    except Exception as error:
        update_job(job_id, status="failed", logs=log[-160:], error=str(error))


class SPAHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def log_message(self, format, *args):
        print("[HTTP]", format % args)

    def end_headers(self):
        # The SPA changes frequently while a user is uploading and training.
        # Do not let a stale index page hide newly deployed controls.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, value, status=HTTPStatus.OK):
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise ApiError("请求内容过大")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except (UnicodeDecodeError, ValueError):
            raise ApiError("JSON 内容无效")

    def read_form(self):
        if int(self.headers.get("Content-Length", "0")) > MAX_UPLOAD_BYTES:
            raise ApiError("上传超过 1GB 限制", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        return cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path_only = parsed.path
            if path_only == "/api/health":
                return self.send_json({"ok": True, "service": "training-dispatch"})
            if path_only == "/api/knowledge":
                return self.send_json({"entries": read_json(KNOWLEDGE / "index.json", [])})
            if path_only == "/api/board":
                config = board_config()
                return self.send_json({"host": config.get("host", ""), "user": config.get("user", "elf")})
            if path_only == "/api/board/queue":
                board_id = parse_qs(parsed.query).get("board_id", ["elf2-board"])[0]
                item = next((candidate for candidate in read_json(QUEUE, []) if candidate.get("board_id") == board_id and candidate.get("status") == "queued"), None)
                if not item:
                    return self.send_json({"item": None})
                manifest = {key: value for key, value in item.items() if key not in ("artifact",)}
                manifest["download_url"] = f"/api/board/artifacts/{item['id']}"
                return self.send_json({"item": manifest})
            match = re.fullmatch(r"/api/board/artifacts/([a-f0-9-]{36})", path_only)
            if match:
                item = queue_item(match.group(1))
                artifact = ROOT / item["artifact"] if item else None
                if not item or not artifact.is_file():
                    raise ApiError("发布文件不存在", HTTPStatus.NOT_FOUND)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(artifact.stat().st_size))
                self.send_header("Content-Disposition", f'attachment; filename="{item["filename"]}"')
                self.end_headers()
                with artifact.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile)
                return
            match = re.fullmatch(r"/api/yolo/jobs/([a-f0-9-]{36})", path_only)
            if match:
                job = read_json(job_path(match.group(1)), None)
                if not job:
                    raise ApiError("任务不存在", HTTPStatus.NOT_FOUND)
                return self.send_json(job)
            path = self.translate_path(self.path)
            if not os.path.exists(path) and not os.path.splitext(path)[1]:
                self.path = "/index.html"
            return super().do_GET()
        except ApiError as error:
            return self.send_json({"error": error.message}, error.status)

    def do_POST(self):
        try:
            path_only = urlparse(self.path).path
            if path_only == "/api/board":
                return self.send_json(board_config(self.read_json_body()))
            if path_only == "/api/knowledge":
                form = self.read_form()
                title = str(form.getfirst("title", "")).strip()
                content = str(form.getfirst("content", "")).strip()
                file_item = form["file"] if "file" in form else None
                if not title or (not content and not getattr(file_item, "filename", "")):
                    raise ApiError("请提供资料标题和正文或文本文件")
                entry_id = str(uuid.uuid4())
                filename = f"{entry_id}.md"
                if getattr(file_item, "filename", ""):
                    suffix = Path(file_item.filename).suffix.lower()
                    if suffix not in (".md", ".txt"):
                        raise ApiError("知识库仅接受 .md 或 .txt 文件")
                    content = file_item.file.read().decode("utf-8", errors="replace")
                (KNOWLEDGE / filename).write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
                entry = {"id": entry_id, "title": title, "filename": filename, "created_at": time.time()}
                index = read_json(KNOWLEDGE / "index.json", [])
                index.insert(0, entry)
                write_json(KNOWLEDGE / "index.json", index)
                return self.send_json({"entry": entry}, HTTPStatus.CREATED)
            match = re.fullmatch(r"/api/knowledge/([a-f0-9-]{36})/publish", path_only)
            if match:
                entry = next((item for item in read_json(KNOWLEDGE / "index.json", []) if item["id"] == match.group(1)), None)
                if not entry:
                    raise ApiError("资料不存在", HTTPStatus.NOT_FOUND)
                payload = self.read_json_body()
                config = board_config(payload)
                content = (KNOWLEDGE / entry["filename"]).read_text(encoding="utf-8")
                knowledge_payload = {
                    "id": entry["id"],
                    "title": entry["title"],
                    "keywords": title_keywords(entry["title"]),
                    "content": content,
                }
                source = ARTIFACTS / f"knowledge-{entry['id']}.json"
                write_json(source, knowledge_payload)
                artifact_name = safe_name(entry["title"], f"knowledge-{entry['id']}")
                item = enqueue_artifact(
                    "knowledge", source, f"{artifact_name}.json",
                    restart_service=bool(payload.get("restart_service", True)),
                    board_host=config.get("host", ""),
                )
                return self.send_json({"ok": True, "queue_id": item["id"], "status": "queued"})
            match = re.fullmatch(r"/api/board/queue/([a-f0-9-]{36})/ack", path_only)
            if match:
                payload = self.read_json_body()
                status = payload.get("status")
                if status not in ("delivered", "failed"):
                    raise ApiError("同步状态无效")
                item = update_queue(match.group(1), status=status, detail=str(payload.get("detail", ""))[:500])
                if not item:
                    raise ApiError("同步任务不存在", HTTPStatus.NOT_FOUND)
                return self.send_json({"ok": True})
            if path_only == "/api/yolo/datasets":
                form = self.read_form()
                raw_items = form["files"] if "files" in form else []
                items = raw_items if isinstance(raw_items, list) else [raw_items]
                if not items:
                    raise ApiError("请上传数据集文件")
                dataset_id = str(uuid.uuid4())
                target = DATASETS / dataset_id
                target.mkdir()
                try:
                    valid_items = [item for item in items if getattr(item, "filename", "")]
                    if len(valid_items) == 1 and Path(valid_items[0].filename).suffix.lower() == ".zip":
                        count = unpack_dataset_zip(valid_items[0], target)
                    else:
                        if len(valid_items) > MAX_DATASET_FILES:
                            raise ApiError("数据集文件数量超过限制")
                        count = 0
                        for item in valid_items:
                            relative = safe_relative(item.filename)
                            output = target / relative
                            output.parent.mkdir(parents=True, exist_ok=True)
                            with output.open("wb") as handle:
                                shutil.copyfileobj(item.file, handle)
                            count += 1
                    yaml = locate_dataset_yaml(target)
                    dataset = {
                        "id": dataset_id,
                        "file_count": count,
                        "data_yaml": str(yaml.relative_to(target)),
                        "root": str(yaml.parent.relative_to(target)),
                    }
                    write_json(target / "dataset.json", dataset)
                    return self.send_json({"dataset": dataset}, HTTPStatus.CREATED)
                except Exception:
                    shutil.rmtree(target, ignore_errors=True)
                    raise
            if path_only == "/api/yolo/jobs":
                payload = self.read_json_body()
                dataset_id = payload.get("dataset_id", "")
                if not re.fullmatch(r"[a-f0-9-]{36}", dataset_id):
                    raise ApiError("数据集编号无效")
                dataset_store = DATASETS / dataset_id
                dataset = read_json(dataset_store / "dataset.json", {})
                relative_yaml = safe_relative(str(dataset.get("data_yaml", "data.yaml")))
                data_yaml = dataset_store / relative_yaml
                if not data_yaml.is_file():
                    raise ApiError("数据集缺少 data.yaml，无法开始训练。")
                dataset_dir = data_yaml.parent
                epochs = int(payload.get("epochs", 80)); image_size = int(payload.get("image_size", 640))
                if not 1 <= epochs <= 1000 or image_size not in (640, 960):
                    raise ApiError("训练参数不在允许范围内")
                job = {"id": str(uuid.uuid4()), "status": "queued", "logs": ["[Queue] 任务已创建。"], "dataset_id": dataset_id, "created_at": time.time(), "epochs": epochs, "image_size": image_size, "model": "yolov5s.pt", "artifact_rknn": None, "progress": {"stage": "waiting_gpu", "current_epoch": 0, "total_epochs": epochs, "percent": 0}}
                write_json(job_path(job["id"]), job)
                threading.Thread(target=run_training, args=(job["id"], dataset_dir, job), daemon=True).start()
                return self.send_json({"job": job}, HTTPStatus.ACCEPTED)
            match = re.fullmatch(r"/api/yolo/jobs/([a-f0-9-]{36})/publish", path_only)
            if match:
                job = read_json(job_path(match.group(1)), None)
                if not job:
                    raise ApiError("任务不存在", HTTPStatus.NOT_FOUND)
                if not job.get("artifact_rknn"):
                    raise ApiError("当前任务没有 RKNN 成品。best.pt 不能直接替换板端 NPU 模型。", HTTPStatus.CONFLICT)
                config = board_config(self.read_json_body())
                class_names = read_json(ROOT / job["artifact_labels"], []) if job.get("artifact_labels") else []
                item = enqueue_artifact(
                    "yolov5-rknn", ROOT / job["artifact_rknn"], "yolov5s-640-640.rknn",
                    board_host=config.get("host", ""), class_names=class_names,
                )
                return self.send_json({"ok": True, "queue_id": item["id"], "status": "queued"})
            raise ApiError("接口不存在", HTTPStatus.NOT_FOUND)
        except ApiError as error:
            return self.send_json({"error": error.message}, error.status)
        except Exception as error:
            print("[API]", error)
            return self.send_json({"error": "服务器处理失败"}, HTTPStatus.INTERNAL_SERVER_ERROR)


if __name__ == "__main__":
    print(f"Serving on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), SPAHandler).serve_forever()

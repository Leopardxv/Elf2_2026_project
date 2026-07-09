#!/usr/bin/env python3
"""
BCI Web Dashboard — Flask SSE + MJPEG + LLM Chat
Real-time EEG waveforms, attention, multi-modal emotion, camera feed, AI assistant.
"""
import sys, os, time, json, socket, threading, io, argparse
from collections import deque
import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, Response, request, render_template_string, jsonify
from scipy.signal import resample

# ============================================================
# 配置
# ============================================================
HW_EXTRACT = [0, 1, 2, 3, 8, 9, 10, 11]
MODEL_REORDER = [1, 0, 3, 2, 4, 5, 6, 7]
CH_LABELS = ['FP1', 'F7', 'F8', 'FP2', 'T5', 'O1', 'O2', 'T6']
CH_COLORS = ['#00ff88', '#00ccff', '#ff8800', '#ff4444',
             '#88ff00', '#ff00ff', '#ffff00', '#00ffff']

FOCUS_PORT = 12345
EEG_PORT = 12346
SAMPLE_RATE = 125
MODEL_INPUT_SAMPLES = 200
WINDOW_SAMPLES = 5 * SAMPLE_RATE  # 5 seconds for waveform display
SSE_RATE = 10  # Hz
WAVEFORM_DOWNSAMPLE = 200  # downsample to 200 points for display
EMOTION_INTERVAL = 1.5

# ============================================================
# 线程安全共享状态
# ============================================================
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.attention_value = 0.5
        self.attention_status = '等待数据...'
        self.emotion_pos = 0.0
        self.emotion_neu = 1.0
        self.emotion_neg = 0.0
        self.eeg_active = False
        self.yolo_active = False
        self.waveform = np.zeros((8, WAVEFORM_DOWNSAMPLE), dtype=np.float32)
        self.waveform_updated = False
        self.camera_jpeg = None
        self.chat_history = []

        # per-modality raw probs: [neg, neu, pos] after reorder
        self.eeg_probs = None
        self.yolo_probs = None

        # EEG buffer for NPU
        self.buf_8 = deque(maxlen=MODEL_INPUT_SAMPLES)

    def set_attention(self, v):
        with self.lock:
            self.attention_value = float(v)
            if v < 0.3:
                self.attention_status = '放松中'
            elif v < 0.7:
                self.attention_status = '普通'
            else:
                self.attention_status = '高度专注!'

    def set_eeg_probs(self, probs):
        """probs: [neg, neu, pos] normalized"""
        with self.lock:
            self.eeg_probs = np.array(probs, dtype=np.float64)
            self.eeg_active = True

    def set_yolo_probs(self, probs):
        """probs: [neg, neu, pos] normalized"""
        with self.lock:
            self.yolo_probs = np.array(probs, dtype=np.float64)
            self.yolo_active = True

    def set_emotion(self, neg, neu, pos, eeg_ok=False, yolo_ok=False):
        with self.lock:
            self.emotion_neg = float(neg)
            self.emotion_neu = float(neu)
            self.emotion_pos = float(pos)
            self.eeg_active = eeg_ok
            self.yolo_active = yolo_ok

    def set_waveform(self, data_8ch):
        """data_8ch: (N, 8), hardware order columns"""
        if len(data_8ch) < 2:
            return
        n = min(len(data_8ch), WINDOW_SAMPLES)
        d = data_8ch[-n:]  # latest N samples
        # downsample to WAVEFORM_DOWNSAMPLE points
        if len(d) > WAVEFORM_DOWNSAMPLE:
            indices = np.linspace(0, len(d) - 1, WAVEFORM_DOWNSAMPLE, dtype=int)
            d = d[indices]
        else:
            # pad with last value
            pad = np.tile(d[-1:], (WAVEFORM_DOWNSAMPLE - len(d), 1))
            d = np.vstack([d, pad]) if len(d) > 0 else np.zeros((WAVEFORM_DOWNSAMPLE, 8))
        with self.lock:
            self.waveform = d.T.astype(np.float32)  # (8, N)
            self.waveform_updated = True

    def set_camera(self, jpeg_bytes):
        with self.lock:
            self.camera_jpeg = jpeg_bytes

    def push_eeg_sample(self, sample_8ch):
        """sample_8ch: (8,) array in hardware order"""
        with self.lock:
            self.buf_8.append(np.array(sample_8ch, dtype=np.float32))

    def get_eeg_inference_data(self):
        """Returns (200, 8) if buffer full, else None"""
        with self.lock:
            if len(self.buf_8) < MODEL_INPUT_SAMPLES:
                return None
            return np.array(list(self.buf_8), dtype=np.float32)

    def get_snapshot(self):
        with self.lock:
            return {
                'attention': self.attention_value,
                'attention_status': self.attention_status,
                'emotion_neg': self.emotion_neg,
                'emotion_neu': self.emotion_neu,
                'emotion_pos': self.emotion_pos,
                'eeg_active': self.eeg_active,
                'yolo_active': self.yolo_active,
            }


state = SharedState()

# ============================================================
# UDP 接收线程
# ============================================================
class UDPReceiver(threading.Thread):
    def __init__(self, port, handler):
        super().__init__(daemon=True)
        self.port = port
        self.handler = handler
        self._run = True

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        try:
            sock.bind(('0.0.0.0', self.port))
        except OSError:
            print(f'[WARN] UDP:{self.port} port busy')
            return
        sock.settimeout(0.3)
        buf = b''
        while self._run:
            try:
                data, _ = sock.recvfrom(16384)
                buf += data
                while True:
                    idx = buf.find(b'\r\n')
                    if idx < 0:
                        break
                    line, buf = buf[:idx], buf[idx + 2:]
                    try:
                        msg = json.loads(line.decode('utf-8'))
                        self.handler(msg)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
            except socket.timeout:
                continue
            except Exception as e:
                print(f'[UDP:{self.port}] {e}')
                break
        sock.close()

    def stop(self):
        self._run = False


def handle_focus(msg):
    if msg.get('type') == 'focus':
        state.set_attention(msg['data'])


def handle_eeg(msg):
    if msg.get('type') not in ('timeSeriesFilt', 'timeSeriesRaw'):
        return
    raw = msg['data']
    try:
        arr = np.array(raw, dtype=np.float32)
    except Exception:
        return
    if arr.ndim != 2 or arr.shape[0] != 16:
        return

    nsamp = arr.shape[1]
    # extract 8ch
    ch8 = arr[HW_EXTRACT, :]  # (8, N)

    # push to NPU buffer
    for t in range(nsamp):
        state.push_eeg_sample(ch8[:, t])

    # update waveform display buffer
    all16 = arr.T  # (N, 16)
    hw8 = all16[:, HW_EXTRACT]  # (N, 8)
    state.set_waveform(hw8)


# ============================================================
# EEG NPU 情绪推理
# ============================================================
def npu_emotion_loop():
    rknn = None

    _save_log_levels = None
    try:
        import logging
        _save_log_levels = dict(logging._nameToLevel)
    except Exception:
        pass

    try:
        from rknnlite.api import RKNNLite

        # rknnlite corrupts logging._nameToLevel — restore it
        if _save_log_levels:
            logging._nameToLevel.clear()
            logging._nameToLevel.update(_save_log_levels)

        rknn = RKNNLite()
        ret = rknn.load_rknn(
            os.path.join(PROJECT_ROOT, 'emotions/EEG-Conformer/eeg_conformer.rknn'))
        if ret != 0:
            print(f'[NPU] Load failed: {ret}')
            return
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print(f'[NPU] Init failed: {ret}')
            return
        print('[NPU] EEG-Conformer ready')
    except ImportError:
        print('[NPU] rknn-toolkit-lite2 not installed')
        return
    except Exception as e:
        print(f'[NPU] Init error: {e}')
        return

    ch_mean = None
    ch_std = None

    while True:
        data = state.get_eeg_inference_data()
        if data is None:
            time.sleep(0.1)
            continue

        # channel reorder: hardware → model order
        reorder = data[:, MODEL_REORDER]  # (200, 8)
        d = reorder.T.astype(np.float64)  # (8, 200)

        # resample 125Hz → 200Hz
        d = resample(d, MODEL_INPUT_SAMPLES, axis=1)

        # z-score with EMA
        if ch_mean is None:
            ch_mean = d.mean(axis=1)
            ch_std = d.std(axis=1) + 1e-8
        alpha = 0.01
        ch_mean = (1 - alpha) * ch_mean + alpha * d.mean(axis=1)
        ch_std = (1 - alpha) * ch_std + alpha * d.std(axis=1)

        d = (d - ch_mean[:, np.newaxis]) / ch_std[:, np.newaxis]
        inp = d[np.newaxis, np.newaxis, :, :].astype(np.float32)

        try:
            out = rknn.inference(inputs=[inp])
            logits = out[1][0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            # model output: [neutral, positive, negative]
            # reorder to: [negative, neutral, positive]
            reordered = np.array([probs[2], probs[0], probs[1]], dtype=np.float64)
            state.set_eeg_probs(reordered)
        except Exception as e:
            print(f'[NPU] Inference error: {e}')

        time.sleep(EMOTION_INTERVAL)


# ============================================================
# 摄像头 + YOLO 线程
# ============================================================
def camera_loop():
    import cv2
    os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
    os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')
    os.environ.setdefault('OPENCV_VIDEOIO_LOG_LEVEL', '0')
    os.environ.setdefault('OPENCV_VIDEOIO_DEBUG', '0')

    _stderr_fd = os.dup(2)
    _null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_null_fd, 2)
    _camera_loop_inner()
    os.dup2(_stderr_fd, 2)
    os.close(_null_fd)
    os.close(_stderr_fd)


def _camera_loop_inner():
    import cv2
    yolo = None
    cap = None

    # rknnlite corrupts logging._nameToLevel — ensure YOLO import has a clean logging env
    _save_levels = None
    try:
        import logging
        _save_levels = dict(logging._nameToLevel)
    except Exception:
        pass

    try:
        from yolo.inference import YoloEmotionRecognizer
        yolo = YoloEmotionRecognizer(device='cpu')
        print('[YOLO] Recognizer ready')
    except Exception as e:
        import traceback
        print(f'[YOLO] Init failed: {e}')
        traceback.print_exc(file=sys.stdout)
    finally:
        if _save_levels:
            logging._nameToLevel.clear()
            logging._nameToLevel.update(_save_levels)

    # try specified camera, fallback to 0
    cameras = [CAMERA_ID] if CAMERA_ID == 0 else [CAMERA_ID, 0]
    for cam_id in cameras:
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if cap.isOpened():
            print(f'[Camera] Opened device {cam_id}')
            break
        cap.release()
        cap = None

    if cap is None:
        print('[Camera] No camera available')
        return

    skip = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        skip += 1
        if yolo and skip % 4 == 0:
            try:
                probs = yolo.get_primary_emotion(frame)
                if probs is not None:
                    # YOLO returns: [positive, neutral, negative]
                    # reorder to: [negative, neutral, positive]
                    reordered = np.array([probs[2], probs[1], probs[0]], dtype=np.float64)
                    state.set_yolo_probs(reordered)
            except Exception:
                pass

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        state.set_camera(buf.tobytes())
        time.sleep(0.05)  # ~20fps


# ============================================================
# 多模态情绪融合循环
# ============================================================
def fuse_loop():
    from yolo.fusion import fuse
    print('[Fusion] Multi-modal emotion fusion ready')
    while True:
        with state.lock:
            eeg = state.eeg_probs
            yolo = state.yolo_probs
            eeg_ok = state.eeg_active
            yolo_ok = state.yolo_active

        if eeg is not None or yolo is not None:
            fused = fuse(eeg, yolo)
            state.set_emotion(fused[0], fused[1], fused[2],
                              eeg_ok=eeg_ok, yolo_ok=yolo_ok)

        time.sleep(EMOTION_INTERVAL)


# ============================================================
# LLM Chat 引擎
# ============================================================
llama_engine = None
llama_lock = threading.Lock()
LLM_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/qwen2.5-0.5b-instruct-q2_k.gguf')


def init_llm():
    global llama_engine
    try:
        from dispatch.llama_engine import LlamaEngine
        engine = LlamaEngine(n_threads=4, n_predict=256)
        if engine.load(LLM_MODEL_PATH):
            llama_engine = engine
            print('[LLM] Qwen2.5-0.5B ready')
        else:
            print('[LLM] Failed to load model')
    except Exception as e:
        print(f'[LLM] Init error: {e}')


# ============================================================
# Flask 应用
# ============================================================
app = Flask(__name__)

# ---- SSE 实时数据推送 ----
@app.route('/api/sse')
def sse_stream():
    def generate():
        last_attention = None
        last_emotion = None
        last_wave_updated = False

        while True:
            snap = state.get_snapshot()

            # attention event
            att = snap['attention']
            if att != last_attention:
                last_attention = att
                yield f"event: attention\ndata: {json.dumps({'value': float(att), 'status': snap['attention_status']})}\n\n"

            # emotion event
            emo = (snap['emotion_neg'], snap['emotion_neu'], snap['emotion_pos'])
            if emo != last_emotion:
                last_emotion = emo
                yield f"event: emotion\ndata: {json.dumps({'neg': float(emo[0]), 'neu': float(emo[1]), 'pos': float(emo[2]), 'eeg_active': snap['eeg_active'], 'yolo_active': snap['yolo_active']})}\n\n"

            # waveform event (only when updated)
            if state.waveform_updated:
                state.waveform_updated = False
                with state.lock:
                    wf = state.waveform.tolist()  # list of 8 lists
                yield f"event: eeg\ndata: {json.dumps({'labels': CH_LABELS, 'data': [[round(v, 2) for v in ch] for ch in wf]})}\n\n"

            time.sleep(1.0 / SSE_RATE)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no',
                             'Connection': 'keep-alive'})


# ---- MJPEG 摄像头流 ----
@app.route('/stream/camera')
def camera_stream():
    def generate():
        while True:
            with state.lock:
                jpeg = state.camera_jpeg
            if jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            else:
                time.sleep(0.1)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---- 最新摄像头单帧 (JPEG) ----
@app.route('/api/camera')
def camera_snapshot():
    with state.lock:
        jpeg = state.camera_jpeg
    if jpeg:
        return Response(jpeg, mimetype='image/jpeg')
    return Response(status=404)


# ---- AI Chat ----
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'missing message'}), 400

    message = data['message'].strip()
    if not message:
        return jsonify({'error': 'empty message'}), 400

    if llama_engine is None:
        return jsonify({'response': '[AI 引擎未加载, 请确保模型文件存在]'})

    with llama_lock:
        try:
            snap = state.get_snapshot()
            ctx = (f'当前用户状态: 注意力={snap["attention"]:.2f}({snap["attention_status"]}), '
                   f'消极={snap["emotion_neg"]*100:.0f}% 中性={snap["emotion_neu"]*100:.0f}% '
                   f'积极={snap["emotion_pos"]*100:.0f}%')
            system = (
                '你是一个矿山应急自救辅助AI助手,当前正通过脑电设备监测用户的注意力和情绪状态。'
                '请根据用户的问题提供简洁有用的帮助(200字以内)。'
            )
            full_prompt = f'{ctx}\n用户: {message}'
            res = llama_engine.chat(full_prompt, system_prompt=system)
            return jsonify({'response': res})
        except Exception as e:
            return jsonify({'response': f'[AI错误: {e}]'})


# ---- 主页 HTML ----
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BCI Web Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI','PingFang SC',sans-serif;background:#111;color:#ccc;overflow:hidden;height:100vh}
#app{display:grid;grid-template-columns:320px 1fr 280px;grid-template-rows:1fr;height:100vh;gap:6px;padding:6px}
.panel{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;overflow:hidden;display:flex;flex-direction:column}
.panel-title{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;padding:6px 10px;border-bottom:1px solid #2a2a2a;flex-shrink:0}
/* LEFT */
#left-panel{display:flex;flex-direction:column;gap:6px}
#left-panel .panel{flex:1}
/* CAMERA */
#cam-container{position:relative;background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden;min-height:200px}
#cam-container img{width:100%;height:100%;object-fit:contain}
#cam-placeholder{color:#444;font-size:14px}
/* CHAT */
#chat-messages{flex:1;overflow-y:auto;padding:8px}
.chat-msg{margin-bottom:8px;padding:6px 10px;border-radius:6px;font-size:13px;line-height:1.5;max-width:100%}
.chat-msg.user{background:#1a3a2a;color:#88ffbb;align-self:flex-end}
.chat-msg.assistant{background:#1a1a2e;color:#aaccff}
.chat-msg.system{background:#2a2a1a;color:#ccaa66;font-size:11px;text-align:center}
#chat-input-area{display:flex;gap:4px;padding:8px;border-top:1px solid #2a2a2a}
#chat-input{flex:1;background:#222;border:1px solid #333;color:#ccc;padding:8px 10px;border-radius:4px;font-size:13px;outline:none}
#chat-input:focus{border-color:#00ff88}
#chat-send{background:#00ff88;color:#111;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;font-weight:bold;font-size:13px}
#chat-send:disabled{opacity:0.4;cursor:default}
/* CENTER */
#wave-canvas{flex:1;display:block}
/* RIGHT */
#right-panel{display:flex;flex-direction:column;gap:6px}
#right-panel .panel{flex:1}
/* ATTENTION */
#att-container{display:flex;flex-direction:column;align-items:center;padding:15px}
#att-gauge{width:180px;height:180px}
#att-value{font-size:42px;font-weight:bold;margin-top:-20px}
#att-status{font-size:16px;margin-top:4px}
/* EMOTION */
#emo-bars{padding:12px 15px}
.emo-row{display:flex;align-items:center;margin-bottom:10px}
.emo-label{width:40px;font-size:12px;text-align:right;margin-right:8px}
.emo-bar-outer{flex:1;height:20px;background:#222;border-radius:3px;overflow:hidden}
.emo-bar-inner{height:100%;border-radius:3px;transition:width 0.3s}
.emo-val{width:42px;font-size:12px;text-align:right;margin-left:8px}
#emo-sources{text-align:center;font-size:10px;color:#555;margin-top:8px}
/* STATUS BAR */
#status-bar{position:fixed;bottom:0;left:0;right:0;height:22px;background:#0a0a0a;border-top:1px solid #222;display:flex;align-items:center;padding:0 10px;font-size:10px;color:#555;z-index:99}
#status-bar span{margin-right:20px}
</style>
</head>
<body>
<div id="app">
  <!-- LEFT: Camera + Chat -->
  <div id="left-panel">
    <div class="panel">
      <div class="panel-title">Camera</div>
      <div id="cam-container">
        <span id="cam-placeholder">摄像头未连接</span>
      </div>
    </div>
    <div class="panel" style="flex:1.5">
      <div class="panel-title">AI Chat</div>
      <div id="chat-messages"></div>
      <div id="chat-input-area">
        <input id="chat-input" type="text" placeholder="向AI助手提问..." onkeydown="if(event.key==='Enter')sendChat()">
        <button id="chat-send" onclick="sendChat()">发送</button>
      </div>
    </div>
  </div>

  <!-- CENTER: EEG Waveforms -->
  <div class="panel">
    <div class="panel-title">EEG Waveforms (8ch)</div>
    <canvas id="wave-canvas"></canvas>
  </div>

  <!-- RIGHT: Attention + Emotion -->
  <div id="right-panel">
    <div class="panel">
      <div class="panel-title">Attention / Focus</div>
      <div id="att-container">
        <svg id="att-gauge" viewBox="0 0 200 200">
          <defs>
            <linearGradient id="att-grad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0" stop-color="#00ff88"/><stop offset="0.5" stop-color="#88ff00"/>
              <stop offset="1" stop-color="#ff4444"/>
            </linearGradient>
          </defs>
          <circle cx="100" cy="100" r="80" fill="none" stroke="#222" stroke-width="14"/>
          <circle id="att-arc" cx="100" cy="100" r="80" fill="none" stroke="url(#att-grad)"
                  stroke-width="14" stroke-dasharray="0 502" stroke-linecap="round"
                  transform="rotate(135 100 100)" style="transition:stroke-dasharray 0.3s"/>
          <circle cx="100" cy="100" r="72" fill="none" stroke="#333" stroke-width="1"/>
        </svg>
        <div id="att-value" style="color:#00ff88">--</div>
        <div id="att-status" style="color:#888">等待数据...</div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">Emotion (多模态融合)</div>
      <div id="emo-bars"></div>
      <div id="emo-sources">模态: EEG ○  YOLO ○</div>
    </div>
  </div>
</div>

<div id="status-bar">
  <span id="st-status">初始化中...</span>
  <span id="st-fps"></span>
</div>

<script>
// ============ EEG Waveform Canvas ============
const waveCanvas = document.getElementById('wave-canvas');
const waveCtx = waveCanvas.getContext('2d');
const chLabels = ['FP1','F7','F8','FP2','T5','O1','O2','T6'];
const chColors = ['#00ff88','#00ccff','#ff8800','#ff4444','#88ff00','#ff00ff','#ffff00','#00ffff'];
let eegData = null;

function resizeWaveCanvas() {
  const rect = waveCanvas.parentElement.getBoundingClientRect();
  waveCanvas.width = rect.width - 2;
  waveCanvas.height = rect.height - 28;
  drawWaveforms();
}

function drawWaveforms() {
  const w = waveCanvas.width, h = waveCanvas.height;
  if (w <= 0 || h <= 0) return;
  waveCtx.clearRect(0, 0, w, h);

  const chH = h / 8;
  for (let i = 0; i < 8; i++) {
    const y0 = i * chH + 3;
    const mid = y0 + chH / 2;

    waveCtx.strokeStyle = '#1a1a1a';
    waveCtx.lineWidth = 1;
    waveCtx.beginPath();
    waveCtx.moveTo(0, y0 + chH);
    waveCtx.lineTo(w, y0 + chH);
    waveCtx.stroke();

    waveCtx.fillStyle = chColors[i];
    waveCtx.font = '10px monospace';
    waveCtx.textBaseline = 'middle';
    waveCtx.fillText(chLabels[i], 4, y0 + 10);

    if (eegData && eegData[i]) {
      const arr = eegData[i];
      const n = arr.length;
      if (n >= 2) {
        const scale = (chH - 6) / 400;
        waveCtx.strokeStyle = chColors[i];
        waveCtx.lineWidth = 1.2;
        waveCtx.beginPath();
        for (let j = 0; j < n; j++) {
          const x = (j / (n - 1)) * w;
          const y = mid - arr[j] * scale;
          if (j === 0) waveCtx.moveTo(x, y);
          else waveCtx.lineTo(x, y);
        }
        waveCtx.stroke();
      }
    }
  }
}

window.addEventListener('resize', () => { resizeWaveCanvas(); drawWaveforms(); });

// ============ Emotion Bars ============
const emoBars = document.getElementById('emo-bars');
const emotions = [
  {label:'消极',color:'#4455ff',id:'neg'},
  {label:'中性',color:'#666666',id:'neu'},
  {label:'积极',color:'#ff4444',id:'pos'},
];
let barEls = [];
emotions.forEach((e,i) => {
  const row = document.createElement('div');
  row.className = 'emo-row';
  row.innerHTML = `<span class="emo-label" style="color:${e.color}">${e.label}</span>
    <div class="emo-bar-outer"><div class="emo-bar-inner" id="bar-${e.id}" style="background:${e.color};width:${i===1?'100':'0'}%"></div></div>
    <span class="emo-val" id="val-${e.id}">${i===1?'100':'0'}%</span>`;
  emoBars.appendChild(row);
  barEls.push(row);
});

// ============ Camera ============
const camContainer = document.getElementById('cam-container');
const camPlaceholder = document.getElementById('cam-placeholder');
const camImg = document.createElement('img');
camImg.style.display = 'none';
camContainer.appendChild(camImg);

fetch('/api/camera').then(r => {
  if (r.ok) {
    camPlaceholder.style.display = 'none';
    camImg.style.display = 'block';
    camImg.src = '/stream/camera?' + Date.now();
    camImg.onerror = () => {
      camImg.style.display = 'none';
      camPlaceholder.style.display = 'block';
      camPlaceholder.textContent = '摄像头断开';
    };
    camImg.onload = () => {
      camImg.onerror = null;
    };
  }
}).catch(() => {});

// ============ SSE Events ============
const attArc = document.getElementById('att-arc');
const attVal = document.getElementById('att-value');
const attSt = document.getElementById('att-status');
const emoSrc = document.getElementById('emo-sources');
const stStatus = document.getElementById('st-status');
const stFps = document.getElementById('st-fps');

const CIRC = 2 * Math.PI * 80; // ~502.65

function updateGauge(v) {
  const dashLen = v * CIRC;
  attArc.setAttribute('stroke-dasharray', `${dashLen} ${CIRC - dashLen}`);
  attVal.textContent = v.toFixed(2);

  if (v < 0.3) attVal.style.color = '#00ff88';
  else if (v < 0.7) attVal.style.color = '#88ff00';
  else attVal.style.color = '#ff4444';
}

function updateEmotion(neg, neu, pos, eegOk, yoloOk) {
  document.getElementById('bar-neg').style.width = (neg*100)+'%';
  document.getElementById('bar-neu').style.width = (neu*100)+'%';
  document.getElementById('bar-pos').style.width = (pos*100)+'%';
  document.getElementById('val-neg').textContent = Math.round(neg*100)+'%';
  document.getElementById('val-neu').textContent = Math.round(neu*100)+'%';
  document.getElementById('val-pos').textContent = Math.round(pos*100)+'%';
  emoSrc.textContent = '模态: EEG '+(eegOk?'●':'○')+'  YOLO '+(yoloOk?'●':'○');
}

let frameCount = 0;
setInterval(() => {
  stFps.textContent = 'UI: ' + frameCount + 'fps';
  frameCount = 0;
}, 1000);

const evtSource = new EventSource('/api/sse');
evtSource.addEventListener('attention', e => {
  const d = JSON.parse(e.data);
  updateGauge(d.value);
  attSt.textContent = d.status;
  stStatus.textContent = 'SSE已连接 | 注意力: ' + d.value.toFixed(2);
  frameCount++;
});
evtSource.addEventListener('emotion', e => {
  const d = JSON.parse(e.data);
  updateEmotion(d.neg, d.neu, d.pos, d.eeg_active, d.yolo_active);
  frameCount++;
});
evtSource.addEventListener('eeg', e => {
  const d = JSON.parse(e.data);
  eegData = d.data;
  drawWaveforms();
  frameCount++;
});
evtSource.onerror = () => { stStatus.textContent = 'SSE断开, 尝试重连...'; };

// ============ Chat ============
let chatPending = false;

function addChatMsg(text, role) {
  const msgs = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  div.textContent = text;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function sendChat() {
  if (chatPending) return;
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;

  addChatMsg(text, 'user');
  input.value = '';
  chatPending = true;
  document.getElementById('chat-send').disabled = true;

  fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text}),
  })
  .then(r => r.json())
  .then(d => {
    addChatMsg(d.response || '[空响应]', 'assistant');
    chatPending = false;
    document.getElementById('chat-send').disabled = false;
  })
  .catch(e => {
    addChatMsg('[网络错误] ' + e, 'system');
    chatPending = false;
    document.getElementById('chat-send').disabled = false;
  });
}

// init
window.addEventListener('load', () => {
  resizeWaveCanvas();
  document.getElementById('chat-input').focus();
});
</script>
</body>
</html>'''


CAMERA_ID = 21
SERVER_PORT = 8765
NO_LLM = False


# ============================================================
# 启动
# ============================================================
def main():
    global CAMERA_ID, SERVER_PORT, NO_LLM
    parser = argparse.ArgumentParser(description='BCI Web Dashboard')
    parser.add_argument('-p', '--port', type=int, default=8765, help='HTTP port (default: 8765)')
    parser.add_argument('-c', '--camera', type=int, default=21, help='Camera device ID (default: 21)')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM model loading')
    args = parser.parse_args()

    CAMERA_ID = args.camera
    SERVER_PORT = args.port
    NO_LLM = args.no_llm

    print('========================================')
    print('  BCI Web Dashboard Starting...')
    print(f'  http://localhost:{SERVER_PORT}')
    print(f'  EEG UDP: ports {FOCUS_PORT}, {EEG_PORT}')
    print(f'  Camera:  /dev/video{CAMERA_ID}')
    print(f'  SSE: /api/sse  |  MJPEG: /stream/camera  |  Chat: /api/chat')
    if NO_LLM:
        print(f'  LLM:     DISABLED (--no-llm)')
    print('========================================')

    # start UDP receivers
    udp_focus = UDPReceiver(FOCUS_PORT, handle_focus)
    udp_eeg = UDPReceiver(EEG_PORT, handle_eeg)
    udp_focus.start()
    udp_eeg.start()

    # start camera
    threading.Thread(target=camera_loop, daemon=True).start()

    # start NPU emotion inference
    threading.Thread(target=npu_emotion_loop, daemon=True).start()

    # start multi-modal fusion
    threading.Thread(target=fuse_loop, daemon=True).start()

    # load LLM in background (large model, slow to load)
    if not NO_LLM:
        threading.Thread(target=init_llm, daemon=True).start()

    app.run(host='0.0.0.0', port=SERVER_PORT, threaded=True, debug=False)


if __name__ == '__main__':
    main()

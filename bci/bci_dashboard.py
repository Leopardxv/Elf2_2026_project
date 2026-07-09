#!/usr/bin/env python3
"""
BCI 实时仪表盘 - OpenBCI Cyton+Daisy (16ch) → 8ch 有效电极
注意力检测 + 情绪识别 (EEG + YOLO 多模态融合) + 8ch 波形展示
"""
import sys, os, time, json, socket, threading
from collections import deque
import numpy as np
import cv2

# 添加 communication 模块路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from scipy.signal import resample
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from yolo.inference import YoloEmotionRecognizer
from yolo.fusion import fuse
from ros2_bridge import ROS2Bridge

# ============================================================
# 通道映射 (对应约束文档 CONSTRAINTS.md)
# 硬件 16ch 数组索引: CH1=FP1, CH2=F7, CH3=F8, CH4=FP2, CH5-8=空, CH9=T5, CH10=O1, CH11=O2, CH12=T6, CH13-16=空
# 提取: [0,1,2,3,8,9,10,11] → 硬件序 [FP1,F7,F8,FP2,T5,O1,O2,T6]
# 模型训练序: [F7,FP1,FP2,F8,T5,O1,O2,T6] → 重排 [1,0,3,2,4,5,6,7]
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
WINDOW_SAMPLES = 5 * SAMPLE_RATE
WAVEFORM_MS = 33
EMOTION_MS = 1500


# ----- UDP 接收 (线程安全) -----
class UDPReceiver(QtCore.QThread):
    data_ready = QtCore.pyqtSignal(object)

    def __init__(self, port):
        super().__init__()
        self.port = port
        self._run = True
        self.packets = 0

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        try:
            sock.bind(('0.0.0.0', self.port))
        except OSError:
            print(f'[WARN] UDP:{self.port} 被占用')
            return
        sock.settimeout(0.3)
        buf = b''
        while self._run:
            try:
                data, _ = sock.recvfrom(16384)
                buf += data
                self.packets += 1
                while True:
                    idx = buf.find(b'\r\n')
                    if idx < 0:
                        break
                    line, buf = buf[:idx], buf[idx + 2:]
                    try:
                        msg = json.loads(line.decode('utf-8'))
                        self.data_ready.emit(msg)
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


# ----- 环形缓冲 (线程安全) -----
class RingBuffer:
    def __init__(self, nch, maxlen=2000):
        self.nch = nch
        self.lock = threading.Lock()
        self.buf = deque(maxlen=maxlen)

    def push(self, frame):
        with self.lock:
            self.buf.append(np.array(frame, dtype=np.float32))

    def tail(self, n=None):
        with self.lock:
            if not self.buf:
                return np.zeros((0, self.nch), dtype=np.float32)
            arr = np.array(list(self.buf), dtype=np.float32)
        if n and len(arr) > n:
            return arr[-n:]
        return arr


# ----- 波形面板 (8 通道, 匹配 OpenBCI GUI) -----
class WaveformPanel(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.setBackground('k')
        self.curves = []
        self.val_texts = []  # 当前电压值显示

        for i, (label, color) in enumerate(zip(CH_LABELS, CH_COLORS)):
            p = self.addPlot(row=i, col=0)
            p.setLabel('left', label, color=color, size='10pt')
            p.showAxis('bottom', i == 7)
            if i < 7:
                p.getAxis('bottom').setStyle(showValues=False)
            p.hideButtons()
            p.setMouseEnabled(x=False, y=False)
            p.setClipToView(True)
            p.setYRange(-200, 200, padding=0)
            p.setXRange(0, WINDOW_SAMPLES)
            p.disableAutoRange()

            c = p.plot(pen=pg.mkPen(color=color, width=1.5))
            self.curves.append(c)

    def update_data(self, data_8ch):
        """data_8ch: (N, 8), 列对应 CH_LABELS 顺序"""
        if len(data_8ch) < 2:
            return
        x = np.arange(len(data_8ch))
        for i in range(8):
            self.curves[i].setData(x, data_8ch[:, i])


# ----- 注意力面板 -----
class FocusPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        l = QtWidgets.QVBoxLayout(self)
        l.setContentsMargins(10, 5, 10, 5)
        t = QtWidgets.QLabel('注意力 Focus')
        t.setStyleSheet('color:#aaa;font-size:14pt;font-weight:bold')
        t.setAlignment(QtCore.Qt.AlignCenter)
        l.addWidget(t)
        self.val = QtWidgets.QLabel('--')
        self.val.setStyleSheet('color:#00ff88;font-size:36pt;font-weight:bold')
        self.val.setAlignment(QtCore.Qt.AlignCenter)
        l.addWidget(self.val)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(24)
        self.bar.setStyleSheet(
            "QProgressBar{border:1px solid #333;border-radius:4px;background:#111}"
            "QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #00ff88,stop:0.5 #88ff00,stop:1 #ff4444);border-radius:3px}")
        l.addWidget(self.bar)
        self.status = QtWidgets.QLabel('等待数据...')
        self.status.setStyleSheet('color:#888;font-size:12pt')
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        l.addWidget(self.status)
        l.addStretch()

    def set_value(self, v):
        v = float(v)
        self.val.setText(f'{v:.2f}')
        self.bar.setValue(int(v * 100))
        if v < 0.3:
            self.status.setText('放松中')
            self.status.setStyleSheet('color:#00ccff;font-size:12pt')
        elif v < 0.7:
            self.status.setText('普通')
            self.status.setStyleSheet('color:#88ff00;font-size:12pt')
        else:
            self.status.setText('高度专注!')
            self.status.setStyleSheet('color:#ff4444;font-size:12pt;font-weight:bold')


# ----- 情绪面板 (支持多模态显示) -----
class EmotionPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        l = QtWidgets.QVBoxLayout(self)
        l.setContentsMargins(10, 5, 10, 5)
        t = QtWidgets.QLabel('情绪 Emotion (多模态融合)')
        t.setStyleSheet('color:#aaa;font-size:14pt;font-weight:bold')
        t.setAlignment(QtCore.Qt.AlignCenter)
        l.addWidget(t)
        self.bars = []
        for name, color in [('消极', '#4444ff'), ('中性', '#888888'), ('积极', '#ff4444')]:
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat(f'{name}  %v%')
            bar.setStyleSheet(
                f"QProgressBar{{border:1px solid #333;border-radius:3px;"
                f"background:#111;color:#fff;font-size:11pt;text-align:left}}"
                f"QProgressBar::chunk{{background:{color};border-radius:2px}}")
            l.addWidget(bar)
            self.bars.append(bar)
        self.mod_label = QtWidgets.QLabel('模态: EEG ○  YOLO ○')
        self.mod_label.setStyleSheet('color:#666;font-size:9pt')
        self.mod_label.setAlignment(QtCore.Qt.AlignCenter)
        l.addWidget(self.mod_label)
        l.addStretch()

    def set_probs(self, fused, eeg=None, yolo=None):
        for i, p in enumerate(fused):
            self.bars[i].setValue(int(p * 100))
        parts = []
        parts.append('EEG ●' if eeg is not None else 'EEG ○')
        parts.append('YOLO ●' if yolo is not None else 'YOLO ○')
        self.mod_label.setText('  '.join(parts))


# ----- 主窗口 -----
class BCIDashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('BCI 实时仪表盘 (EEG + YOLO 多模态)')
        self.setStyleSheet('background-color:#1a1a1a')
        self.resize(1100, 800)

        # 数据缓冲: 16ch 全通道 + 8ch 有效通道
        self.buf_16 = RingBuffer(16)
        self.buf_8 = RingBuffer(8)
        self._samples = 0
        self._rate_t0 = time.time()

        # NPU
        self._rknn = None
        self._init_npu()

        # YOLO 视觉 (后台加载，不阻塞 GUI)
        self._yolo = None
        self._yolo_probs = None
        self._yolo_lock = threading.Lock()
        self._yolo_cap = None
        self._yolo_frame = None
        self._yolo_running = True
        self._yolo_ready = threading.Event()
        threading.Thread(target=self._init_yolo, daemon=True).start()

        # 二进制协议发送器
        try:
            from communication.elf_sender import ElfSender
            self.sender = ElfSender('localhost', 5566)
        except Exception as e:
            print(f'[WARN] ElfSender 未启用: {e}')
            self.sender = None

        # ROS2 桥接 (发布注意力, 订阅速度)
        try:
            self.ros2 = ROS2Bridge()
            self.ros2.velocity_updated.connect(self._on_robot_velocity)
            print('[ROS2] 桥接已就绪')
        except Exception as e:
            print(f'[WARN] ROS2 桥接未启用: {e}')
            self.ros2 = None

        self._build_ui()

        # UDP
        self.udp_focus = UDPReceiver(FOCUS_PORT)
        self.udp_eeg = UDPReceiver(EEG_PORT)
        self.udp_focus.data_ready.connect(self._on_focus)
        self.udp_eeg.data_ready.connect(self._on_eeg)
        self.udp_focus.start()
        self.udp_eeg.start()

        # 定时器
        self._timer_wave = QtCore.QTimer()
        self._timer_wave.timeout.connect(self._update_wave)
        self._timer_wave.start(WAVEFORM_MS)

        self._timer_emo = QtCore.QTimer()
        self._timer_emo.timeout.connect(self._run_emotion)
        self._timer_emo.start(EMOTION_MS)

        self._timer_rate = QtCore.QTimer()
        self._timer_rate.timeout.connect(self._show_rate)
        self._timer_rate.start(2000)

    def _init_yolo(self):
        try:
            self._yolo = YoloEmotionRecognizer(device='cpu')
            self._yolo_cap = cv2.VideoCapture(21)
            self._yolo_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._yolo_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self._yolo_cap.isOpened():
                print('[WARN] 摄像头 21 无法打开, 尝试 0')
                self._yolo_cap = cv2.VideoCapture(0)
            if self._yolo_cap.isOpened():
                print('[YOLO] 摄像头已启动')
                self._yolo_ready.set()
                self._yolo_loop()
            else:
                print('[WARN] 无可用摄像头, YOLO 仅作回退')
                self._yolo_cap = None
                self._yolo_ready.set()
        except Exception as e:
            print(f'[WARN] YOLO 初始化失败: {e}')
            self._yolo = None
            self._yolo_cap = None
            self._yolo_ready.set()

    def _yolo_loop(self):
        skip_frames = 3
        frame_count = 0
        while self._yolo_running and self._yolo_cap and self._yolo_cap.isOpened():
            ok, frame = self._yolo_cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            frame_count += 1
            if frame_count % (skip_frames + 1) != 0:
                continue
            try:
                probs = self._yolo.get_primary_emotion(frame)
                with self._yolo_lock:
                    self._yolo_probs = probs
                    self._yolo_frame = frame.copy()
            except Exception:
                pass

    def _init_npu(self):
        try:
            from rknnlite.api import RKNNLite
            self._rknn = RKNNLite()
            ret = self._rknn.load_rknn(
                '/home/elf/Projects/emotions/EEG-Conformer/eeg_conformer.rknn')
            if ret != 0:
                print(f'[ERROR] RKNN load: {ret}')
                return
            ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
            if ret != 0:
                print(f'[ERROR] RKNN init: {ret}')
                return
            print('[NPU] OK')
        except ImportError:
            print('[WARN] rknn-toolkit-lite2 未安装')
        except Exception as e:
            print(f'[ERROR] NPU: {e}')

    def _build_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        ml = QtWidgets.QVBoxLayout(cw)
        ml.setContentsMargins(5, 5, 5, 5)
        ml.setSpacing(4)

        # 波形
        self.wave = WaveformPanel()
        ml.addWidget(self.wave, stretch=6)

        # 底部
        bot = QtWidgets.QHBoxLayout()
        self.focus = FocusPanel()
        bot.addWidget(self.focus, stretch=3)
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setStyleSheet('color:#333')
        bot.addWidget(sep)
        self.emotion = EmotionPanel()
        bot.addWidget(self.emotion, stretch=3)
        ml.addLayout(bot, stretch=4)

        self._status = QtWidgets.QLabel('等待 UDP 数据...')
        self._status.setStyleSheet('color:#666;font-size:10pt')
        ml.addWidget(self._status)

    # ---- UDP 回调 ----
    def _on_focus(self, msg):
        if msg.get('type') == 'focus':
            self.focus.set_value(msg['data'])
            if self.sender:
                self.sender.send_attention(msg['data'])
            if self.ros2:
                self.ros2.publish_attention(msg['data'])

    def _on_eeg(self, msg):
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
        self._samples += nsamp

        # 写入 16ch 缓冲 (用于波形显示 - 后续提取 8ch)
        for t in range(nsamp):
            self.buf_16.push(arr[:, t])

        # 提取 8ch 硬件序 → 情绪推理缓冲
        ch8 = arr[HW_EXTRACT, :]  # (8, N)
        for t in range(nsamp):
            self.buf_8.push(ch8[:, t])
            if self.sender:
                self.sender.push_eeg(ch8[:, t])

    def _show_rate(self):
        dt = time.time() - self._rate_t0
        if dt > 0:
            rate = self._samples / dt
            self._status.setText(
                f'UDP 接收: {rate:.0f} 采样/秒 (目标 125Hz) | '
                f'波形缓冲: {len(self.buf_16.buf)} / {self.buf_16.buf.maxlen} | '
                f'UDP包: {self.udp_eeg.packets}')
        self._samples = 0
        self._rate_t0 = time.time()

    def _on_robot_velocity(self, linear, angular):
        if self.sender:
            self.sender.send_robot_velocity(linear, angular)

    # ---- 定时更新 ----
    def _update_wave(self):
        # 从 16ch 全缓冲取数据 → 提取 8ch 硬件序 → 显示
        raw16 = self.buf_16.tail(WINDOW_SAMPLES)  # (N, 16)
        if len(raw16) > 0:
            ch8_hw = raw16[:, HW_EXTRACT]  # (N, 8) 按硬件序
            self.wave.update_data(ch8_hw)

    def _run_emotion(self):
        raw = self.buf_8.tail(MODEL_INPUT_SAMPLES)  # (200, 8) 硬件序
        if len(raw) < MODEL_INPUT_SAMPLES or self._rknn is None:
            eeg_probs = None
        else:
            # 重排: 硬件序 [FP1,F7,F8,FP2,T5,O1,O2,T6] → 模型序 [F7,FP1,FP2,F8,T5,O1,O2,T6]
            reorder = raw[:, MODEL_REORDER]  # (200, 8)

            # 转置为 (8, 200)
            d = reorder.T.astype(np.float64)

            # 重采样 125Hz → 200Hz
            d = resample(d, MODEL_INPUT_SAMPLES, axis=1)

            # 逐通道 Z-score (每 200 步更新一次统计量)
            if not hasattr(self, '_ch_mean'):
                self._ch_mean = d.mean(axis=1)
                self._ch_std = d.std(axis=1) + 1e-8
            # 滚动更新归一化统计量 (EMA)
            alpha = 0.01
            self._ch_mean = (1 - alpha) * self._ch_mean + alpha * d.mean(axis=1)
            self._ch_std = (1 - alpha) * self._ch_std + alpha * d.std(axis=1)

            d = (d - self._ch_mean[:, np.newaxis]) / self._ch_std[:, np.newaxis]
            inp = d[np.newaxis, np.newaxis, :, :].astype(np.float32)

            try:
                out = self._rknn.inference(inputs=[inp])
                logits = out[1][0]
                eeg_probs = np.exp(logits) / np.sum(np.exp(logits))
            except Exception:
                eeg_probs = None

        with self._yolo_lock:
            yolo_probs = self._yolo_probs

        fused = fuse(eeg_probs, yolo_probs)
        self.emotion.set_probs(fused.tolist(), eeg=eeg_probs, yolo=yolo_probs)

        if self.sender:
            self.sender.send_emotion(*fused.tolist())

    def closeEvent(self, ev):
        self._timer_wave.stop()
        self._timer_emo.stop()
        self._timer_rate.stop()
        self.udp_focus.stop()
        self.udp_eeg.stop()
        self._yolo_running = False
        if self._yolo_cap:
            self._yolo_cap.release()
        if self.ros2:
            self.ros2.shutdown()
        if self.sender:
            self.sender.stop()
        if self._rknn:
            self._rknn.release()
        ev.accept()


def main():
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(26, 26, 26))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(200, 200, 200))
    app.setPalette(pal)
    dash = BCIDashboard()
    dash.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

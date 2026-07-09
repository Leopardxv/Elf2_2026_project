#!/usr/bin/env python3
"""
二进制协议 TCP 发送线程 — 供 bci_dashboard 调用
处理所有消息类型的编码、缓冲和异步发送
"""

import time
import struct
import socket
import threading
import queue
from typing import Optional

from elf_protocol import (
    encode_attention, encode_emotion, encode_eeg, encode_heartbeat,
    encode_robot_velocity, TYPE_ATTENTION, TYPE_EMOTION, TYPE_EEG
)


class ElfSender:
    """ELF2 二进制协议 TCP 发送器"""

    def __init__(self, host='localhost', port=5566, heartbeat_interval=1.5):
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval

        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = True
        self._sock = None

        # EEG 缓冲
        self._eeg_buf = []

        # 未连接时的暂存队列
        self._pending = []

        self._last_send_time = time.time()
        self._connected = threading.Event()

        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()
        print(f'[ElfSender] 目标: {host}:{port}')

    def _next_id(self):
        with self._lock:
            fid = self._frame_id
            self._frame_id = (self._frame_id + 1) & 0xFFFF
            return fid

    def send_attention(self, score: float):
        """发送注意力 (0x01)"""
        frame = encode_attention(self._next_id(), score)
        self._enqueue(frame)

    def send_emotion(self, positive: float, neutral: float, negative: float):
        """发送情绪 (0x03)"""
        frame = encode_emotion(self._next_id(), positive, neutral, negative)
        self._enqueue(frame)

    def send_robot_velocity(self, linear: float, angular: float):
        """发送小车速度 (0x02)"""
        frame = encode_robot_velocity(self._next_id(), linear, angular)
        self._enqueue(frame)

    def push_eeg(self, sample_8ch: list):
        """缓存一个采样点 (8 个通道值)"""
        with self._lock:
            self._eeg_buf.append([float(v) for v in sample_8ch])

    def _enqueue(self, frame: bytes):
        """发送或暂存"""
        self._last_send_time = time.time()
        if self._connected.is_set() and self._sock:
            try:
                self._sock.sendall(frame)
                return
            except (BrokenPipeError, ConnectionResetError, OSError):
                self._connected.clear()
        # 暂存，最多 200 帧
        if len(self._pending) < 200:
            self._pending.append(frame)

    def _send_eeg_burst(self):
        """将缓存的 EEG 采样点打包成一帧发送"""
        with self._lock:
            if not self._eeg_buf:
                return
            data = self._eeg_buf[:]
            self._eeg_buf.clear()

        if not data:
            return

        channels = len(data[0])
        samples = len(data)
        frame = encode_eeg(self._next_id(), channels, samples, data)
        self._enqueue(frame)

    def _send_loop(self):
        """主发送循环"""
        last_eeg_flush = time.time()

        while self._running:
            try:
                # 建立连接
                if self._sock is None:
                    self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    self._sock.settimeout(1)
                    self._sock.connect((self.host, self.port))
                    self._connected.set()
                    print(f'[ElfSender] 已连接 {self.host}:{self.port}')
                    # 发送暂存帧
                    while self._pending:
                        try:
                            self._sock.sendall(self._pending.pop(0))
                        except Exception:
                            break

                # 定时刷新 EEG 缓冲 (~100ms)
                now = time.time()
                if now - last_eeg_flush > 0.1:
                    self._send_eeg_burst()
                    last_eeg_flush = now

                # 心跳
                if now - self._last_send_time > self.heartbeat_interval:
                    frame = encode_heartbeat(self._next_id())
                    self._enqueue(frame)

                time.sleep(0.05)

            except (ConnectionRefusedError, ConnectionResetError,
                    BrokenPipeError, socket.timeout, OSError) as e:
                self._connected.clear()
                if self._sock:
                    self._sock.close()
                self._sock = None
                print(f'[ElfSender] 连接断开，重试中... ({e})')
                time.sleep(1)
            except Exception as e:
                print(f'[ElfSender] 错误: {e}')
                self._connected.clear()
                if self._sock:
                    self._sock.close()
                self._sock = None
                time.sleep(1)

    def stop(self):
        """停止发送"""
        self._running = False
        if self._sock:
            self._sock.close()
        print('[ElfSender] 已停止')

#!/usr/bin/env python3
"""
ELF2 二进制协议 TCP 接收服务 — 单文件部署版
功能：监听 TCP 5566 端口，接收 ELF2 开发板发来的二进制帧，
      解析各类型消息，写入 JSONL 日志文件。
用法：python3 receiver.py              # 前台运行
      nohup python3 receiver.py &      # 后台运行

协议说明：见 README.md / 约束文档
依赖：无 (仅 Python 3 标准库)
"""

import os
import sys
import json
import time
import struct
import socket
import signal
import threading
import zlib
from datetime import datetime

# ============================================================
# 协议常量 (与 elf_protocol.py 完全一致)
# ============================================================
SYNC_MAGIC = 0xAA55
HEADER_SIZE = 17       # 2+2+1+4+8
CRC_SIZE = 4
MAX_PAYLOAD = 64 * 1024

TYPE_HEARTBEAT = 0x00
TYPE_ATTENTION = 0x01
TYPE_ROBOT = 0x02
TYPE_EMOTION = 0x03
TYPE_EEG = 0x04

TYPE_NAMES = {0x00: 'HB', 0x01: 'ATT', 0x02: 'ROBOT', 0x03: 'EMO', 0x04: 'EEG'}

LISTEN_HOST = '0.0.0.0'
LISTEN_PORT = 5566
LOG_DIR = './logs'

stats = {'frames': 0, 'bad_crc': 0, 'bad_len': 0, 'bytes': 0,
         'start': time.time(), 'by_type': dict.fromkeys(range(5), 0)}


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def log_decoded(msg_type, decoded, timestamp_ms):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.fromtimestamp(timestamp_ms / 1000)
    path = os.path.join(LOG_DIR, f'eeg_{ts.strftime("%Y%m%d_%H")}.jsonl')
    record = {
        'ts': timestamp_ms,
        'time': ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        'type': TYPE_NAMES.get(msg_type, f'0x{msg_type:02X}'),
        'data': decoded,
    }
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ============================================================
# Payload 解码
# ============================================================
def decode_payload(msg_type, payload):
    try:
        if msg_type == TYPE_ATTENTION:
            return {'attention': round(struct.unpack('>f', payload)[0], 6)}
        elif msg_type == TYPE_ROBOT:
            lv, av = struct.unpack('>ff', payload)
            return {'linear': round(lv, 4), 'angular': round(av, 4)}
        elif msg_type == TYPE_EMOTION:
            pos, neu, neg = struct.unpack('>fff', payload)
            return {'positive': round(pos, 4), 'neutral': round(neu, 4),
                    'negative': round(neg, 4)}
        elif msg_type == TYPE_EEG:
            ch = payload[0]
            samples = struct.unpack('>H', payload[1:3])[0]
            if len(payload) < 3 + ch * samples * 4:
                return {'channels': ch, 'samples': samples, 'error': 'payload_truncated'}
            eeg_flat = struct.unpack(f'>{ch * samples}i',
                                      payload[3:3 + ch * samples * 4])
            ch_means = []
            for c in range(ch):
                vals = [eeg_flat[s * ch + c] for s in range(samples)]
                ch_means.append(round(sum(vals) / len(vals), 2))
            ch_mins = [round(min(eeg_flat[s * ch + c] for s in range(samples)), 2)
                       for c in range(ch)]
            ch_maxs = [round(max(eeg_flat[s * ch + c] for s in range(samples)), 2)
                       for c in range(ch)]
            return {'channels': ch, 'samples': samples,
                    'ch_means': ch_means, 'ch_mins': ch_mins, 'ch_maxs': ch_maxs}
        elif msg_type == TYPE_HEARTBEAT:
            return {'heartbeat': True}
    except (struct.error, IndexError):
        return None
    return None


# ============================================================
# 帧解码
# ============================================================
def decode_frame(data: bytes):
    if len(data) < HEADER_SIZE + CRC_SIZE:
        return None
    sync = struct.unpack('>H', data[0:2])[0]
    if sync != SYNC_MAGIC:
        return None
    try:
        _, frame_id, msg_type, payload_len, timestamp_ms = \
            struct.unpack('>HHB I Q', data[:HEADER_SIZE])
    except struct.error:
        return None
    if payload_len > MAX_PAYLOAD:
        stats['bad_len'] += 1
        return None
    expected = HEADER_SIZE + payload_len + CRC_SIZE
    if len(data) < expected:
        return None
    payload = data[HEADER_SIZE:HEADER_SIZE + payload_len]
    received_crc = struct.unpack('>I', data[HEADER_SIZE + payload_len:expected])[0]
    if crc32(data[:HEADER_SIZE + payload_len]) != received_crc:
        stats['bad_crc'] += 1
        return None
    return {
        'frame_id': frame_id,
        'msg_type': msg_type,
        'timestamp_ms': timestamp_ms,
        'payload': payload,
    }


# ============================================================
# TCP 客户端处理
# ============================================================
def handle_client(conn, addr):
    buf = b''
    last_stats = time.time()
    print(f'[+] {addr[0]}:{addr[1]} 已连接')

    while True:
        try:
            data = conn.recv(4096)
            if not data:
                break
            buf += data
            stats['bytes'] += len(data)

            while True:
                idx = buf.find(b'\xaa\x55')
                if idx < 0:
                    buf = buf[-1:] if len(buf) > 0 and buf[-1] == 0xAA else b''
                    break
                if idx > 0:
                    buf = buf[idx:]
                if len(buf) < HEADER_SIZE + CRC_SIZE:
                    break
                try:
                    payload_len = struct.unpack('>I', buf[5:9])[0]
                except struct.error:
                    buf = buf[2:]
                    continue
                if payload_len > MAX_PAYLOAD:
                    buf = buf[2:]
                    stats['bad_len'] += 1
                    continue
                total = HEADER_SIZE + payload_len + CRC_SIZE
                if len(buf) < total:
                    break
                frame_data = buf[:total]
                buf = buf[total:]

                decoded = decode_frame(frame_data)
                if decoded is None:
                    stats['bad_crc'] += 1
                    continue

                stats['frames'] += 1
                stats['by_type'][decoded['msg_type']] += 1

                payload = decode_payload(decoded['msg_type'], decoded['payload'])
                if decoded['msg_type'] != TYPE_HEARTBEAT:
                    log_decoded(decoded['msg_type'], payload, decoded['timestamp_ms'])

            if time.time() - last_stats > 60:
                print_stats()
                last_stats = time.time()

        except socket.timeout:
            continue
        except Exception as e:
            print(f'[!] {e}')
            break

    conn.close()
    print(f'[-] {addr[0]}:{addr[1]} 断开')


def print_stats(signum=None, frame=None):
    elapsed = time.time() - stats['start']
    fps = stats['frames'] / elapsed if elapsed > 0 else 0
    kbps = stats['bytes'] / 1024 / elapsed if elapsed > 0 else 0
    tc = ' '.join(f'{TYPE_NAMES[t]}:{stats["by_type"][t]}' for t in range(5))
    print(f'[Stats] {elapsed:.0f}s | {stats["frames"]}帧({fps:.1f}/s) | '
          f'{stats["bytes"]/1024:.1f}KB | CRC错:{stats["bad_crc"]} '
          f'超长:{stats["bad_len"]} | {tc}')


def main():
    print(f'[ELF2 Receiver] 监听 {LISTEN_HOST}:{LISTEN_PORT}')
    print(f'[ELF2 Receiver] 日志: {os.path.abspath(LOG_DIR)}/')
    print(f'[ELF2 Receiver] 按 Ctrl+C 停止')
    signal.signal(signal.SIGINT, lambda s, f: (print_stats(), sys.exit(0)))
    signal.signal(signal.SIGUSR1, print_stats)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((LISTEN_HOST, LISTEN_PORT))
    srv.listen(5)

    while True:
        conn, addr = srv.accept()
        t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        t.start()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
二进制通信协议 完整测试
1. 单元测试: 编码→解码往返验证
2. 真实数据测试: 用 S1_session1 数据集模拟真实发送
3. TCP 端到端测试: 本地收发
"""
import sys, os, time, struct, json, threading, socket
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elf_protocol import (
    encode_frame, decode_frame,
    encode_attention, encode_emotion, encode_eeg, encode_heartbeat,
    encode_robot_velocity,
    TYPE_HEARTBEAT, TYPE_ATTENTION, TYPE_ROBOT_VELOCITY,
    TYPE_EMOTION, TYPE_EEG, SYNC_MAGIC, HEADER_SIZE, CRC_SIZE
)
from elf_sender import ElfSender

PASS = 0
FAIL = 0


def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  [PASS] {name}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name} {detail}')


def test_encode_decode():
    """往返测试: 编码→解码，验证数据完整性"""
    print('\n=== 1. 编码/解码往返测试 ===')

    # 注意力
    f = encode_attention(42, 0.87)
    d = decode_frame(f)
    check('Attention encode→decode', d and abs(d['payload']['attention'] - 0.87) < 0.0001,
          f'got {d["payload"]["attention"] if d else None}')

    # 情绪
    f = encode_emotion(43, 0.15, 0.42, 0.43)
    d = decode_frame(f)
    check('Emotion encode→decode',
          d and abs(d['payload']['positive'] - 0.15) < 0.0001
          and abs(d['payload']['neutral'] - 0.42) < 0.0001
          and abs(d['payload']['negative'] - 0.43) < 0.0001)

    # 脑电
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]  # 3samples, 3ch
    f = encode_eeg(44, 3, 3, data)
    d = decode_frame(f)
    check('EEG encode→decode', d and d['msg_type'] == TYPE_EEG
          and d['payload']['channels'] == 3
          and d['payload']['samples'] == 3)

    # 心跳
    f = encode_heartbeat(45)
    d = decode_frame(f)
    check('Heartbeat encode→decode',
          d and d['msg_type'] == TYPE_HEARTBEAT
          and d['payload']['heartbeat'])

    # 小车
    f = encode_robot_velocity(46, 0.5, -1.2)
    d = decode_frame(f)
    check('Robot encode→decode',
          d and abs(d['payload']['linear_velocity'] - 0.5) < 0.0001
          and abs(d['payload']['angular_velocity'] - (-1.2)) < 0.0001)


def test_error_detection():
    """测试错误检测"""
    print('\n=== 2. 错误检测测试 ===')

    # 错误魔数
    f = encode_attention(1, 0.5)
    bad = bytearray(f)
    bad[0] = 0xFF
    d = decode_frame(bytes(bad))
    check('Bad sync byte → None', d is None)

    # 错误的 CRC
    bad = bytearray(f)
    bad[-1] ^= 0xFF  # 翻转 CRC 最后字节
    d = decode_frame(bytes(bad))
    check('Bad CRC → None', d is None)

    # 过大的 payload 长度
    bad = bytearray(f)
    struct.pack_into('>I', bad, 5, 100000)  # 改长度字段
    d = decode_frame(bytes(bad))
    check('Oversize payload → None', d is None)

    # 截断帧
    d = decode_frame(f[:10])
    check('Truncated frame → None', d is None)


def test_frame_id_wraparound():
    """测试帧 ID 循环"""
    print('\n=== 3. 帧 ID 循环测试 ===')
    frames = []
    for i in range(65540):
        f = encode_attention(i, 0.5, timestamp_ms=i)
        d = decode_frame(f)
        frames.append(d['frame_id'])
    check('Frame ID wraps around',
          frames[0] == 0 and frames[-1] == 65539 % 65536)


def test_real_eeg_data():
    """用真实数据集测试"""
    print('\n=== 4. 真实 EEG 数据编码/解码测试 ===')
    data_path = '/home/elf/Projects/emotions/S1_session1.npy'
    if not os.path.exists(data_path):
        print('  [SKIP] 数据文件不存在')
        return

    raw = np.load(data_path, allow_pickle=True)
    all_data = []
    for i in range(len(raw)):
        all_data.append(np.array(raw[i]).astype(np.float32))
    X = np.concatenate(all_data, axis=0)  # (N, 8, 200)

    # 取第一批 200 个时间点, X[0].T → (200, 8)
    subset = X[0].T  # (200, 8)
    ch8_data = [[float(subset[s, c]) for c in range(8)] for s in range(200)]

    t0 = time.time()
    f = encode_eeg(100, 8, 200, ch8_data)
    encode_time = (time.time() - t0) * 1000

    t0 = time.time()
    d = decode_frame(f)
    decode_time = (time.time() - t0) * 1000

    frame_size = len(f)
    payload_size = frame_size - HEADER_SIZE - CRC_SIZE

    check('Real EEG encode succeeds', d is not None and d['msg_type'] == TYPE_EEG)
    check(f'Frame size {frame_size}B (payload {payload_size}B)',
          payload_size == 3 + 8 * 200 * 4)
    print(f'  编码耗时: {encode_time:.2f}ms, 解码耗时: {decode_time:.2f}ms')


def test_tcp_end_to_end():
    """TCP 端到端测试"""
    print('\n=== 5. TCP 端到端测试 ===')

    # 启动本地接收服务
    received_frames = []

    def server():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('127.0.0.1', 15566))
        srv.listen(1)
        srv.settimeout(3)
        try:
            conn, addr = srv.accept()
            buf = b''
            start = time.time()
            while time.time() - start < 2:
                try:
                    data = conn.recv(4096)
                    if not data:
                        break
                    buf += data
                except socket.timeout:
                    break
            conn.close()

            # 解析所有帧
            while len(buf) >= HEADER_SIZE + CRC_SIZE:
                sync = struct.unpack('>H', buf[0:2])[0]
                if sync != SYNC_MAGIC:
                    buf = buf[1:]
                    continue
                payload_len = struct.unpack('>I', buf[5:9])[0]
                total = HEADER_SIZE + payload_len + CRC_SIZE
                if len(buf) < total:
                    break
                d = decode_frame(buf[:total])
                if d:
                    received_frames.append(d)
                buf = buf[total:]
        finally:
            srv.close()

    srv_thread = threading.Thread(target=server, daemon=True)
    srv_thread.start()
    time.sleep(0.5)

    # 发送端
    sender = ElfSender('127.0.0.1', 15566, heartbeat_interval=0.3)
    sender.send_attention(0.75)
    sender.send_emotion(0.2, 0.5, 0.3)
    for _ in range(10):
        sender.push_eeg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    time.sleep(1.5)
    sender.stop()
    srv_thread.join(timeout=2)

    types = set(d['msg_type'] for d in received_frames)
    check('Received frames exist', len(received_frames) > 0,
          f'got {len(received_frames)} frames')
    check('Got attention', TYPE_ATTENTION in types)
    check('Got emotion', TYPE_EMOTION in types)
    check('Got EEG', TYPE_EEG in types)


def test_large_eeg():
    """大数据量压力测试"""
    print('\n=== 6. 大数据量测试 ===')
    # 模拟 8 通道 × 500 采样点 (真实场景一帧可能没有这么多，但测试极限)
    data = [[float(i * 8 + c) for c in range(8)] for i in range(500)]
    f = encode_eeg(200, 8, 500, data)
    d = decode_frame(f)

    check('500-sample EEG encodes OK', d is not None and d['msg_type'] == TYPE_EEG)
    check(f'Payload size: {len(f) - HEADER_SIZE - CRC_SIZE}B',
          len(f) - HEADER_SIZE - CRC_SIZE == 3 + 8 * 500 * 4)


def test_protocol():
    print('=' * 60)
    print('ELF2 二进制通信协议 完整测试')
    print('=' * 60)

    test_encode_decode()
    test_error_detection()
    test_frame_id_wraparound()
    test_real_eeg_data()
    test_tcp_end_to_end()
    test_large_eeg()

    print('\n' + '=' * 60)
    total = PASS + FAIL
    print(f'结果: {PASS}/{total} 通过, {FAIL} 失败')
    if FAIL == 0:
        print('协议实现正确 ✓')
    else:
        print('存在未通过项 ✗')
    print('=' * 60)
    return FAIL == 0


if __name__ == '__main__':
    ok = test_protocol()
    sys.exit(0 if ok else 1)

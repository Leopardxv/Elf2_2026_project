#!/usr/bin/env python3
"""
通信测试 — 发送各类型二进制帧到服务器，验证端到端传输
用法:
  1. 确保服务器 receiver.py 在运行
  2. python3 test_send.py
"""
import sys, os, time, socket, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elf_protocol import (
    encode_attention, encode_emotion, encode_eeg, encode_heartbeat,
    encode_robot_velocity, FRAME_OVERHEAD
)

HOST = 'localhost'
PORT = 5566


def show_frame(name, data):
    print(f'  [{name}] {len(data)}B  '
          f'魔数: 0x{data[0]:02X}{data[1]:02X}  '
          f'CRC: 0x{struct.unpack(">I", data[-4:])[0]:08X}')


def main():
    print(f'连接 {HOST}:{PORT}...')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(5)
        sock.connect((HOST, PORT))
    except (ConnectionRefusedError, socket.timeout) as e:
        print(f'连接失败: {e}')
        print('提示: 先运行 SSH 隧道:')
        print('  ssh -p 43189 -R 5566:localhost:5566 root@connect.cqa1.seetacloud.com -N &')
        return

    print('已连接，开始发送测试数据...\n')

    # 1. Attention
    f = encode_attention(1, 0.87)
    show_frame('Attention', f)
    sock.sendall(f)

    # 2. Emotion
    f = encode_emotion(2, 0.25, 0.35, 0.40)
    show_frame('Emotion', f)
    sock.sendall(f)

    # 3. EEG (3 channels × 5 samples, 模拟小数据)
    eeg_data = [
        [100.0, 200.0, 300.0],   # sample 0
        [110.0, 210.0, 310.0],   # sample 1
        [120.0, 220.0, 320.0],   # sample 2
        [130.0, 230.0, 330.0],   # sample 3
        [140.0, 240.0, 340.0],   # sample 4
    ]
    f = encode_eeg(3, 3, 5, eeg_data)
    show_frame('EEG (3ch×5)', f)
    sock.sendall(f)

    # 4. Heartbeat
    f = encode_heartbeat(4)
    show_frame('Heartbeat', f)
    sock.sendall(f)

    # 5. Robot
    f = encode_robot_velocity(5, 0.5, -1.2)
    show_frame('Robot', f)
    sock.sendall(f)

    # 6. 再发几个 Attention 验证连续性
    for i in range(3):
        f = encode_attention(6 + i, round(0.5 + i * 0.1, 2))
        sock.sendall(f)

    print(f'\n共发送 8 帧，等待服务端确认...')
    time.sleep(1)

    sock.close()
    print('测试完成。检查服务器日志:')
    print('  tail -5 /root/elf2_receiver/logs/eeg_*.jsonl')
    print('  (看 receiver.py 终端输出，应有 ATT/EMO/EEG/HB 统计)')


if __name__ == '__main__':
    main()

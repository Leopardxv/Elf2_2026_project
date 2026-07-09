#!/usr/bin/env python3
"""UDP 数据验证工具 - 打印原始接收值，方便和 GUI 对比"""
import socket, json, time, sys
import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 12346
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 5

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
sock.bind(('0.0.0.0', PORT))
sock.settimeout(5)

print(f'监听 UDP:{PORT}，等 {COUNT} 个包...')
print()

for n in range(COUNT):
    try:
        data, addr = sock.recvfrom(16384)
        for line in data.split(b'\r\n'):
            if not line.strip():
                continue
            try:
                msg = json.loads(line.decode('utf-8'))
            except:
                continue
            t = msg.get('type', '?')
            if t not in ('timeSeriesFilt', 'timeSeriesRaw'):
                # focus 包
                if t == 'focus':
                    print(f'[Focus] {msg["data"]:.4f}')
                continue

            arr = np.array(msg['data'], dtype=np.float32)
            print(f'[{t}] shape={arr.shape} ', end='')
            if arr.shape[0] >= 4:
                print(f' CH1: {arr[0,:3]}  CH2: {arr[1,:3]}  CH9: {arr[8,:3]}', end='')
            print()

    except socket.timeout:
        print('超时 - 没有数据。GUI 在推流吗？')
        break

sock.close()
print('\n如果 GUI 刷新生效 (Shift+N 加载配置)，以上 CH1 的值应该和 GUI 波形面板 CH1 一致。')

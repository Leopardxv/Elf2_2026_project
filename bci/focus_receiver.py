#!/usr/bin/env python3
"""接收 OpenBCI GUI 经由 UDP 发送的 Focus 注意力数据（连续值 0.0~1.0）"""

import socket
import json
import time
from collections import deque

LISTEN_PORT = 12345
SMOOTH_SAMPLES = 10

class FocusReceiver:
    def __init__(self, port=LISTEN_PORT, smooth_n=SMOOTH_SAMPLES):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(1.0)
        self.history = deque(maxlen=smooth_n)
        self.count = 0
        self.last_print = 0

    def smoothed(self, val):
        self.history.append(val)
        return sum(self.history) / len(self.history)

    def run(self):
        print(f"Listening for Focus data on UDP port {LISTEN_PORT}...")
        print("Configure OBCI GUI Networking → Focus → UDP → 127.0.0.1:{}\n".format(LISTEN_PORT))

        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
                msg = json.loads(data.decode())
                self.count += 1
                now = time.time()

                val = msg["data"]
                sm = self.smoothed(val)

                if now - self.last_print >= 0.2:
                    bar = "#" * int(val * 40)
                    bar_s = "-" * int(sm * 40)
                    print(f"  raw={val:.4f}  [{bar}]")
                    print(f"  sma={sm:.4f}   [{bar_s}]\n")
                    self.last_print = now

            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break

        self.sock.close()


if __name__ == "__main__":
    FocusReceiver().run()

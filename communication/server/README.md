# ELF2 通信接收服务 — 部署 & 使用

## 概述

接收 ELF2 开发板通过二进制协议发送的脑电(EEG)、注意力(Attention)、情绪(Emotion) 及心跳(Heartbeat) 数据。该服务监听 TCP 5566 端口，解析协议帧，写入按小时分割的 JSONL 日志文件。

### 数据流向

```
ELF2 开发板 (bci_dashboard.py)
    │ 二进制协议帧 (TCP)
    ▼
ELF2 → ssh -L 5566:localhost:5566 → 云服务器
    │
    ▼
receiver.py (监听 0.0.0.0:5566)
    │ 解析帧: Attention / Emotion / EEG / Heartbeat
    ▼
./logs/eeg_YYYYMMDD_HH.jsonl  (每小时一个文件)
```

**关键**：SSH 隧道必须用 `-L`（正向转发），不是 `-R`。

### 协议帧格式（简要）

| 字段 | 字节 | 说明 |
|------|------|------|
| 同步魔数 | 2 | 固定 `0xAA55` (大端) |
| 帧序列号 | 2 | 0~65535 循环 |
| 消息类型 | 1 | 0x00=心跳, 0x01=注意力, 0x03=情绪, 0x04=脑电 |
| 载荷长度 | 4 | Payload 字节数 |
| 时间戳 | 8 | 毫秒级 Unix 时间戳 |
| 数据载荷 | N | 见各消息类型定义 |
| CRC32 | 4 | 帧头+载荷校验 |

完整协议定义见约束文档。

## 部署

**要求**: Python 3.6+, 无需额外依赖。

### 方法 1: nohup (推荐)

```bash
unzip server.zip -d /root/elf2_receiver
cd /root/elf2_receiver
nohup python3 receiver.py > receiver.log 2>&1 &
```

### 方法 2: systemd (有 systemd 的机器)

```bash
cat > /etc/systemd/system/elf2-receiver.service << EOF
[Unit]
Description=ELF2 Binary Protocol Receiver
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/root/elf2_receiver
ExecStart=python3 /root/elf2_receiver/receiver.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now elf2-receiver
```

### 方法 3: screen (无 systemd 环境)

```bash
screen -dmS elf2 python3 /root/elf2_receiver/receiver.py
```

## ELF2 开发板端操作

开发板通过 SSH 正向隧道 (`-L`) 将本地 5566 映射到服务器：

```bash
ssh -p 43189 -L 5566:localhost:5566 root@connect.cqa1.seetacloud.com -N &
```

仪表盘 `bci_dashboard.py` 已集成 `ElfSender`，启动后自动连接 `localhost:5566` 发送二进制协议数据。

## 日志格式

每行一条 JSON，键值：

```json
{
  "ts": 1717692800123,
  "time": "2026-06-06 22:13:20.123",
  "type": "ATT",
  "data": {"attention": 0.870000}
}
```

### 消息类型

| type | 含义 | data 字段 |
|------|------|-----------|
| `HB` | 心跳 | `{"heartbeat": true}` |
| `ATT` | 注意力 | `{"attention": 0.87}` → 0~1 |
| `ROBOT` | 小车速度 | `{"linear": 0.5, "angular": -1.2}` |
| `EMO` | 情绪 | `{"positive": 0.4, "neutral": 0.3, "negative": 0.3}` |
| `EEG` | 脑电 | `{"channels": 8, "samples": 25, "ch_means": [...], "ch_mins": [...], "ch_maxs": [...]}` |

`EEG` 类型只存每个通道的统计摘要 (均值/最小值/最大值)，不存完整波形 (波形数据太大不适合 JSONL)。

## 端到端验证 (2026-06-06)

本地测试通过，二进制协议端到端正常工作：

```
# ELF2 开发板
ssh -L 5566:localhost:5566 root@server -N &
python3 test_send.py

# 服务器端输出
$ cat logs/eeg_*.jsonl
{"ts":...,"type":"ATT","data":{"attention":0.87}}
{"ts":...,"type":"EMO","data":{"positive":0.25,"neutral":0.35,"negative":0.4}}
{"ts":...,"type":"EEG","data":{"channels":3,"samples":5,"ch_means":[120,220,320],...}}
{"ts":...,"type":"ROBOT","data":{"linear":0.5,"angular":-1.2}}
{"ts":...,"type":"ATT","data":{"attention":0.5}}
{"ts":...,"type":"ATT","data":{"attention":0.6}}
{"ts":...,"type":"ATT","data":{"attention":0.7}}
```

## 查看数据

```bash
# 实时查看最新数据
tail -f logs/eeg_*.jsonl

# 按时间段过滤
grep '"type":"EMO"' logs/eeg_20260606_22.jsonl

# 统计今日数据量
wc -l logs/eeg_$(date +%Y%m%d)_*.jsonl
```

## 停止服务

```bash
# nohup 方式
pkill -f receiver.py

# systemd 方式
systemctl stop elf2-receiver

# screen 方式
screen -S elf2 -X quit
```

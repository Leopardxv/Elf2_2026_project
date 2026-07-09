# BCI Web Dashboard 使用说明

## 启动

```bash
conda activate eeg
python /home/elf/Projects/bci/bci_web_server.py
```

浏览器打开 `http://localhost:8765`（局域网内其他设备用 `http://<板子IP>:8765`）。

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-p PORT` | HTTP 端口 | 8765 |
| `-c CAMERA_ID` | 摄像头设备号 | 21 |
| `--no-llm` | 不加载 AI 聊天模型 | 关闭 |

示例：
```bash
# 指定端口 + 跳过 LLM（启动更快）
python /home/elf/Projects/bci/bci_web_server.py -p 8080 --no-llm

# 使用摄像头 0
python /home/elf/Projects/bci/bci_web_server.py -c 0
```

## 关闭

直接在终端 `Ctrl+C`，或：
```bash
pkill -f bci_web_server
```

## 页面功能

| 区域 | 功能 | 数据来源 |
|------|------|----------|
| 左侧 - 摄像头 | 实时画面 MJPEG 流 | USB 摄像头 |
| 左侧 - AI 聊天 | Qwen2.5-0.5B 对话（含 EEG 状态上下文） | CPU LLM |
| 中间 - 波形 | 8 通道 EEG 实时波形 | BrainFlow UDP (端口 12346) |
| 右上 - 注意力 | 圆弧仪表盘 + 状态文字 | BrainFlow UDP (端口 12345) |
| 右下 - 情绪 | 三分类概率条 + 模态来源标注 | NPU EEG + YOLO 融合 |

## 依赖

- 需要先启动 OpenBCI GUI（`bash ~/Projects/bci/start_bci.sh`），数据通过 UDP 传入
- YOLO 情绪识别需要摄像头
- NPU 推理需要 RK3588 板载 NPU
- LLM 需要 `/home/elf/Projects/models/qwen2.5-0.5b-instruct-q2_k.gguf`

## 后端接口

| 路径 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 仪表盘 HTML 页面 |
| `/api/sse` | GET | SSE 实时推送（attention/emotion/eeg 事件） |
| `/stream/camera` | GET | MJPEG 摄像头流 |
| `/api/camera` | GET | 摄像头单帧 JPEG |
| `/api/chat` | POST | AI 聊天（body: `{"message":"..."}` ） |

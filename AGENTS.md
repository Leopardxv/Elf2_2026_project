# AGENTS.md — 嵌赛项目 (ELF2 / RK3588)

**作品：** 基于ELF2（RK3588）与自制脑电装置的矿山应急自救辅助系统
**赛道：** 嵌赛-瑞芯微赛道 | **截止：** 2025年7月初 | **最终提交：** 视频 + 技术文档

## 环境

- **板子：** RK3588 (ELF2) ARM64 — 本项目不使用 x86
- **Python：** **必须用 conda 环境 `eeg`** (`~/miniforge3/envs/eeg`) — Python 3.10
  - 系统 Python 是 3.13，不能运行本项目代码
- **NPU 推理：** `rknn-toolkit-lite2` v2.1.0，NPU 驱动 **0.9.6**（内核内置）
- **ROS2：** Humble，`source /opt/ros/humble/setup.bash`

## 架构概览（7 个子模块）

| 目录 | 功能 | 状态 |
|------|------|------|
| `bci/` | 脑电数据采集 + PyQt5 仪表盘 | ✅ |
| `yolo/` | 面部表情识别 (CPU) | ✅ |
| `emotions/` | EEG-Conformer NPU 推理 | ✅ |
| `communication/` | ELF2↔云端 TCP 二进制协议 | ✅ |
| `xiaoche_ws/` | ROS2 小车双板通信 | ✅ |
| `ai/` | NPU 工具链 | ✅ |
| `dispatch/` | **模型调度：LLM + Vision NPU** | ✅ |

## dispatch 模块 — 三合一模型引擎

```
dispatch/
├── __init__.py          # LlamaEngine, VisionNPU, RKLLMEngine
├── llama_engine.py      # CPU LLM (llama.cpp), 可用
├── rknn_vision.py       # NPU 视觉 (ResNet18 分类), 可用
├── rkllm_engine.py      # NPU LLM (RKLLM), 等驱动升级
└── config.py            # 模型路径
```

### 当前可用模型

| 引擎 | 模型 | 大小 | 用途 |
|------|------|------|------|
| `LlamaEngine` | Qwen2.5-0.5B GGUF | 396MB | CPU 中文对话，25 tok/s |
| `VisionNPU` | ResNet18 NPU | 12MB | NPU 图像分类 |
| `VisionNPU` | YOLOX-S NPU | 11MB | 等驱动升级（需 runtime ≥2.3.2） |
| `RKLLMEngine` | — | — | 等驱动升级（需驱动 ≥0.9.7） |

### 用法

```python
from dispatch import LlamaEngine, VisionNPU

# CPU 对话
llama = LlamaEngine()
llama.load("models/qwen2.5-0.5b-instruct-q2_k.gguf")
llama.chat("矿难后避难硐室首先做什么")

# NPU 分类
vision = VisionNPU()
vision.load_classifier("models/resnet18_for_rk3588.rknn")
vision.classify(frame_bgr, top_k=5)
```

### NPU 驱动现状

- **当前版本**: 0.9.6（内核内置，够常规 RKNN 推理用）
- **升级方法**: 从飞凌资料包取 `~/elf2/OK3588-linux-source/kernel` 源码，改版本号后编译内核，替换 `/boot/Image`
- **升级后解锁**: RKLLM（LLM on NPU）、YOLOX（目标检测 on NPU）、其他自定义算子模型

## 关键约束 (agent 容易犯错的)

### EEG 通道映射 — 每次推理前必须重排

```
HW_EXTRACT = [0, 1, 2, 3, 8, 9, 10, 11]  # 从16ch中提取
MODEL_REORDER = [1, 0, 3, 2, 4, 5, 6, 7]  # 重排为模型训练顺序
# 硬件序: [FP1, F7, F8, FP2, T5, O1, O2, T6]
# 模型序: [F7, FP1, FP2, F8, T5, O1, O2, T6]
```
参见: `bci/CONSTRAINTS.md`

### 注意力值是反向指标

- `attention=0.0` → 高度专注/紧张
- `attention=1.0` → 完全放松/放空
- 来源: BrainFlow `RELAXATION` 指标，**不是**传统注意力
- 小车速度系数应为 `1.0 - attention`（专注时加速）

### EEG 预处理管道 (给 NPU 推理)

```
1. 缓冲 200 采样点 @125Hz
2. 通道重排: data[:, MODEL_REORDER]
3. 重采样: scipy.signal.resample(data, 200, axis=1)  # 125Hz→200Hz
4. 逐通道 Z-score 归一化
5. Shape: [1, 1, 8, 200]  # batch, channel, electrodes, time
6. RKNN 推理 → Softmax → 3 类概率
```

### 通信协议严谨

- 同步头: `0xAA55` (大端)
- 帧结构: Header(17B) + Payload + CRC32(4B)
- CRC32 覆盖 Header + Payload 的**全部字节**
- 心跳间隔: 1.5 秒空闲即发
- 参见: `communication/约束文档.md` 和 `docs/对接文档_服务器端.md`

## 常用命令

```bash
# BCI 一键启动 (含 ARM64 JAR 修补)
bash ~/Projects/bci/start_bci.sh

# BCI 仪表盘
conda activate eeg
python ~/Projects/bci/bci_dashboard.py

# LLM CPU 对话测试
conda activate eeg
python -c "from dispatch import LlamaEngine; e=LlamaEngine(); e.load('models/qwen2.5-0.5b-instruct-q2_k.gguf'); print(e.chat('你好'))"

# NPU 视觉分类测试
conda activate eeg
python -c "from dispatch import VisionNPU; v=VisionNPU(); v.load_classifier('models/resnet18_for_rk3588.rknn'); print(v.classify(frame)[:3])"

# ROS2 小车构建
source /opt/ros/humble/setup.bash
cd ~/Projects/xiaoche_ws && colcon build

# ROS2 运行 (RK3588 端)
source ~/Projects/xiaoche_ws/install/setup.bash
ros2 run xiaoche_communication node_rk3588

# 通信协议测试
conda activate eeg
python ~/Projects/communication/test_protocol.py

# 无常规 test/lint 命令 — 比赛项目，手动验证
```

## 硬件启动顺序

1. 连接 OpenBCI 设备 (USB → `/dev/ttyUSB0` or `/dev/ttyACM0`)
2. `bash start_bci.sh` → 自动修补 JAR (ARM64)、启动 OpenBCI GUI
3. GUI 内: 选 "Cyton (with Daisy)" → START SESSION → 空格启动数据流 → Shift+N 加载网络
4. `python bci_dashboard.py`

# 多模态情绪识别项目

基于 **RK3588 边缘计算平台**，融合 **YOLO 视觉（面部表情）** 与 **脑电信号（EEG）** 实现多模态情绪识别，结果通过 TCP 协议实时上传云端。

## 目录说明

### `yolo/`
YOLO 视觉模块 —— 摄像头实时人脸检测与表情识别，作为多模态情绪识别的视觉输入分支。

### `bci/`
脑机接口数据采集与监控系统。基于 OpenBCI Cyton+Daisy (16 通道) 硬件，通过 UDP 实时传输 EEG 波形和注意力指标，提供 PyQt5 可视化面板，并在 RK3588 NPU 上运行 EEG-Conformer 模型做情绪推理。

### `emotions/`
情绪识别模型训练与部署。基于 EEG-Conformer 架构（卷积 + Transformer），在 SEED 数据集上训练三分类（消极/中性/积极）情绪识别模型，已转换为 RKNN 格式部署到 RK3588 NPU，适配 8 通道 OpenBCI 头环。

### `communication/`
ELF2 板与云端的 TCP 二进制通信协议。定义 5 种消息类型（心跳/注意力/情绪/EEG/小车速度），含 CRC32 校验、断线重连、毫秒级时间戳对齐，云端接收端按小时存储 JSONL 日志。

### `ai/`
Rockchip NPU AI 模型部署工具链。包含 RKNN Toolkit 2（通用模型转换与推理）和 RKLLM（大语言模型部署，如 DeepSeek-R1-Distill-Qwen-1.5B），用于将训练好的模型转换到 RK3588 NPU 上运行。

### `xiaoche_ws/`
ROS2 小车工作空间。双板架构（RK3588 + A733），通过 ROS2 Topic 双向传输注意力系数与速度指令，实现基于注意力检测的车速联动控制。

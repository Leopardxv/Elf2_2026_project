# BCI 实时仪表盘 - 开发约束文档

## 1. 硬件与通道约束

### 脑电帽配置
- **设备**: OpenBCI Cyton + Daisy (16 通道)
- **实际有效电极**: 仅 8 个通道有信号

### 通道映射表

| 电极名 | 10-20 位置 | GUI 通道号 | 16ch 数组索引 | 模型顺序索引 |
|--------|-----------|-----------|-------------|------------|
| FP1 | 额极左 | CH1 | [0] | 1 |
| F7 | 额下左 | CH2 | [1] | 0 |
| F8 | 额下右 | CH3 | [2] | 3 |
| FP2 | 额极右 | CH4 | [3] | 2 |
| T5 | 后颞左 | CH9 | [8] | 4 |
| O1 | 枕左 | CH10 | [9] | 5 |
| O2 | 枕右 | CH11 | [10] | 6 |
| T6 | 后颞右 | CH12 | [11] | 7 |

### 提取与重排规则

```python
# 从 16 通道提取 8 个有效通道
HW_EXTRACT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]

# 硬件顺序: [FP1, F7, F8, FP2, T5, O1, O2, T6]
CH_LABELS_HW = ['FP1', 'F7', 'F8', 'FP2', 'T5', 'O1', 'O2', 'T6']

# 模型训练顺序: [F7, FP1, FP2, F8, T5, O1, O2, T6]
CH_LABELS_MODEL = ['F7', 'FP1', 'FP2', 'F8', 'T5', 'O1', 'O2', 'T6']

# 重排映射: 硬件索引 → 模型索引
MODEL_REORDER = [1, 0, 3, 2, 4, 5, 6, 7]
```

**重要**: 波形显示使用硬件顺序 (`CH_LABELS_HW`)，NPU 推理前必须通过 `MODEL_REORDER` 重排为模型训练顺序。

---

## 2. UDP 通信协议

### 端口分配

| 端口 | 数据类型 | GUI DataType 值 | 用途 |
|------|---------|----------------|------|
| 12345 | Focus | 1 | 注意力数值 (BrainFlow ML 模型) |
| 12346 | TimeSeriesFilt | 4 | 滤波后时间序列 |

### Focus 包格式 (端口 12345)
```json
{"type": "focus", "data": 0.8234}
```
`data` 为浮点数，范围 0.0 ~ 1.0。每秒约 10-20 包。

### 时间序列包格式 (端口 12346)
```json
{"type": "timeSeriesFilt", "data": [[c1_t1, c1_t2, ...], [c2_t1, ...], ..., [c16_t1, ...]]}
```
`data` 为 `[16通道][N采样点/包]` 二维数组，N 通常 10~20。数据已经过 OpenBCI GUI 滤波。

---

## 3. 注意力检测

直接接收 OpenBCI GUI Focus widget 输出，**不自行计算**。

- 算法: BrainFlow 内置 ML 模型 (`BrainFlowMetrics.RELAXATION`)
- 输入: GUI 内选中的 EEG 通道
- 输出: 连续值 0.0 ~ 1.0
- 显示: 进度条 + 数值 + 状态文字

---

## 4. 情绪识别 (EEG-Conformer NPU)

### 预处理管道（严格按顺序）

1. **缓冲**: 累计 200 个采样点（~1.6 秒 @125Hz）
2. **通道重排**: `raw[:, MODEL_REORDER]` (硬件顺序 → 模型训练顺序)
3. **重采样**: `scipy.signal.resample(data, 200, axis=1)` (125Hz → 200Hz)
4. **逐通道 Z-score**: `(x - channel_mean) / (channel_std + 1e-8)`
5. **Shape**: `[1, 1, 8, 200]` (batch × channel × electrodes × time)
6. **NPU 推理**: `rknn.inference(inputs=[inp])`
7. **Softmax**: `exp(logits) / sum(exp(logits))` → 3 类概率

### 模型规格

| 属性 | 值 |
|------|-----|
| 文件 | `eeg_conformer.rknn` |
| 大小 | ~676KB |
| 输入 | `[1, 1, 8, 200]` float32 |
| 输出 | 2 个张量: features `[1, 280]`, logits `[1, 3]` |
| 分类 | 消极 / 中性 / 积极 |
| NPU 核心 | `RKNNLite.NPU_CORE_0` |
| 推理延迟 | ~3ms |

---

## 5. 采样率约束

| 参数 | 值 |
|------|-----|
| Cyton+Daisy 实际采样率 | **125 Hz** |
| 模型训练采样率 | 200 Hz |
| 模型输入时间维度 | 200 points |
| 每包 UDP 采样点数 | ~10-20 |
| 数据缓冲窗口 | 200+ 采样点 |

**重采样说明**: 125Hz 的 EEG 与 200Hz 训练数据的时间尺度不同。使用 `scipy.signal.resample` 将 125Hz 数据重采样到 200 点以匹配模型输入维度。

---

## 6. 界面布局 (上下分屏)

```
┌──────────────────────────────────────────────────────┐
│                  波形面板 (8 通道 × 5 秒)              │
│   FP1 ──╲╱───     硬件顺序显示，30fps 刷新            │
│   F7  ──╲╱──╲╱──  Y 轴 ±200 μV                      │
│   ...                                                │
│   T6  ─╲╱──╲──                                       │
├─────────────────┬──────────┬──────────────────────────┤
│   注意力 Focus   │          │     情绪 Emotion          │
│  [████████░░]   │          │   消极  ████░░ 15%       │
│     0.87        │          │   中性  ██████ 42%       │
│   (专注中)       │          │   积极  ████████ 43%     │
└─────────────────┴──────────┴──────────────────────────┘
```

---

## 7. 性能约束

| 指标 | 目标值 |
|------|--------|
| 波形刷新率 | 30 fps |
| 注意力更新延迟 | < 200ms |
| 情绪推理间隔 | 1.5 秒 |
| NPU 内存占用 | < 1MB |
| 总内存占用 | < 200MB |
| CPU 占用 | < 20% (单核) |

---

## 8. 启动流程

1. 确保 OpenBCI 设备已连接
2. 运行 `start_bci.sh` 启动 OpenBCI GUI
3. 在 GUI 中：
   - 选择数据源 "Cyton (with Daisy)"
   - 点击 "START SESSION"
   - 按空格键启动数据流
   - 按 `Shift+N` 加载网络配置
   - 添加 Focus widget 和 Networking widget
4. 运行 `python3 bci_dashboard.py`
5. 停止时先关仪表盘，再关 GUI

---

## 9. 依赖

```
PyQt5 >= 5.15
pyqtgraph
numpy
scipy
rknn-toolkit-lite2 >= 2.3.0
torch (仅加载权重时使用)
```

---

## 10. 文件清单

| 文件 | 路径 | 用途 |
|------|------|------|
| `bci_dashboard.py` | `~/Projects/bci/` | 主程序 |
| `eeg_conformer.rknn` | `~/Projects/emotions/EEG-Conformer/` | NPU 模型 |
| `best_fold1.pth` | `~/Projects/emotions/` | 权重文件 (参考) |
| `DaisyUserSettings.json` | `~/Projects/bci/OpenBCI_GUI/Settings/` | GUI UDP 配置 |
| `start_bci.sh` | `~/Projects/bci/` | GUI 启动脚本 |

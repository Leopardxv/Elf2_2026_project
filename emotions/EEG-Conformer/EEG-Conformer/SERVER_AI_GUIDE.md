# Server AI Assistant Guide for EEG-Conformer 8-Channel Project
## 8通道自制脑电数据训练 - 服务器 AI 助手执行指南与开发约束

本文件用以指导服务器端具有 GPU 加速环境的 AI 编码助手，帮助其快速理解项目架构、修改模型通道参数，并安全地开展基于 8 通道脑电数据的训练、评估和可视化。

---

## 1. 项目背景与目标 (Project Background)

*   **当前项目核心**：基于 **EEG-Conformer**（结合 CNN + Transformer 的轻量级脑电分类网络）进行脑电信号特征提取与分类。
*   **输入数据变更**：原始项目使用的是 62 通道（SEED数据集）或 22 通道（BCI IV 2a）。**本阶段目标是适配用户自制的 8 通道 OpenBCI 复刻脑电帽数据**。
*   **运行环境**：本代码包将从无 GPU 的本地开发机迁移至**具有 GPU/CUDA 加速能力**的 Linux/Windows 服务器中运行。

---

## 2. 文件结构与核心模块说明 (File Structure)

服务器 AI 助手应知悉以下核心代码文件的分工：
*   `my_conformer_seed.py`：单被试的 5 折交叉验证训练流程脚本，包含 S&R（时域分段重构）数据增强以及训练 Loss / Accuracy 双 Y 轴曲线生成。
*   `my_conformer_seed_visualized.py`：在训练基础上，集成了分批特征提取的 t-SNE 特征分布图生成和 MNE 脑地形图展示。
*   `EEG-Conformer-main/conformer.py`：标准的 EEG-Conformer 模型结构实现。

---

## 3. 服务器 AI 助手核心开发约束 (Development Constraints)

服务器 AI 助手在修改、调试和运行此项目时，**必须严格遵守**以下准则：

### 约束 3.1：GPU/CUDA 动态配置与资源友好
1.  **禁止硬编码固定 GPU 编号**：原生脚本中存在 `os.environ["CUDA_VISIBLE_DEVICES"] = '0'`。在多用户共享的服务器环境中，这可能会导致显卡冲突。应允许通过环境变量指定或动态检测。
2.  **动态设备选择**：在代码初始化阶段增加自动检测，若 CUDA 可用则使用 `cuda`，否则平滑降级至 `cpu`：
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```
3.  **显存 OOM 防范**：进行 t-SNE 特征降维及推理时，不可一次性将全部数据集特征送入 GPU 进行 `forward` 运算。**必须使用分批推理（Mini-batch Inference）**，参考 `my_conformer_seed_visualized.py` 中的 `visualize_tsne` 实现。

### 约束 3.2：模型通道适配修改规则（8通道修改点）
要将默认的 62 通道模型修改为 8 通道，AI 助手只需对模型第一阶段的 `PatchEmbedding` 空间卷积进行调整：
1.  修改 `shallownet` 中的第二层 2D 卷积核高度为 **`8`**：
    ```python
    # 修改前：nn.Conv2d(40, 40, (62, 1), (1, 1))
    # 修改后：
    nn.Conv2d(40, 40, (8, 1), (1, 1))
    ```
2.  **保持其他各层输入输出维度不变**：由于空间卷积核的高度与电极数 $C=8$ 保持一致，经过该卷积后，电极维度会自动压缩至 $1$。因此，后续的自注意力模块（Transformer Encoder）和分类头（ClassificationHead）的扁平化特征维度（默认 280 维）无需进行多余更改，从而保持了原有预训练权重迁移和模型结构的稳定性。
3.  **确定 8 通道电极布局配置**：此自制脑电帽使用前额 4 个电极与后脑勺 4 个电极处于同一圆周水平面的排布方式。电极的国际 10-20 系统标准命名序列必须配置为：
    ```python
    ch_names = ['F7', 'Fp1', 'Fp2', 'F8', 'T5', 'O1', 'O2', 'T6']
    ```
    （注：其中左耳后上方的 T5 和右耳后上方的 T6 在有些新标准中又记作 P7 和 P8，两者指向完全相同的物理位置，在 MNE 库中均支持，脚本内统一采用该序列即可）。

### 约束 3.3：数据准备与预处理规范
1.  用户输入的脑电信号 shape 必须调整为：`[Batch_size, 1, 8, TimeSamples]`（例如，如果采样率为 200Hz，2秒切片，则输入为 `[Batch_size, 1, 8, 400]`）。
2.  在送入网络前，**必须进行基本的滤波和去漂移处理**（建议采用 1-40Hz 带通滤波 + 50Hz 陷波滤波器过滤工频干扰）。
3.  如果训练初期样本数量较少，**必须保留并激活 `augment_snr`（S&R 数据增强）**，以防止 Transformer 模块在 8 通道小数据集上产生严重过拟合。

### 约束 3.4：输出与日志管理
1.  所有生成的训练可视化图（例如：Loss-Acc 曲线 `sub*_fold*_curve.png`、t-SNE 降维分布图 `*_tsne.png`、脑激活地形图 `*_cat.png`）和训练日志文本，**必须统一输出到指定的 `./results/` 文件夹下**，禁止随意生成在根目录。
2.  地形图绘制需确保依赖库 `mne` 已正确安装，并且通道名称与 10-20 系统的 8 通道电极列表 `['F7', 'Fp1', 'Fp2', 'F8', 'T5', 'O1', 'O2', 'T6']` 保持一致，以确保地形图电极空间插值位置的正确渲染。

---

## 4. 推荐服务器部署与运行指令 (Commands Guide)

服务器 AI 助手应使用如下指令进行环境部署和运行：

```bash
# 1. 创建并激活虚拟环境 (以 conda 为例)
conda create -n eeg_env python=3.10 -y
conda activate eeg_env

# 2. 安装核心依赖
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install numpy matplotlib scikit-learn einops mne scipy

# 3. 运行 8 通道模拟测试 (验证环境与修改是否成功)
python my_conformer_seed.py

# 4. 运行可视化训练评估 (生成 t-SNE 散点图与地形图)
python my_conformer_seed_visualized.py
```

# -*- coding: utf-8 -*-
"""
Colab: DeepSeek-R1-1.5B → RKLLM 转换脚本
========================================
对照 PDF "ELF 2 - RK3588本地部署DeepSeek-R1" 第3.1节

使用说明：
  1. 把 DeepSeek-R1-Distill-Qwen-1.5B.zip 上传到 Colab 的 /content/ 目录
  2. 依次运行各 Cell
  3. 最后下载生成的 .rkllm 文件

参考：
  PDF: ELF 2 - RK3588本地部署DeepSeek-R1.pdf
  GitHub: https://github.com/airockchip/rknn-llm
"""

# %% [Cell 1] 安装 RKLLM-Toolkit (PDF 2.2.1 节)
#    注意：必须在 Colab x86 + GPU 环境下运行
#    板上 librkllmrt.so 是 v1.1.4，必须用匹配版本

!pip install rkllm-toolkit==1.1.4 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

import os

MODEL_ZIP = "/content/DeepSeek-R1-Distill-Qwen-1.5B.zip"
MODEL_DIR = "/content/DeepSeek-R1-Distill-Qwen-1.5B"

# %% [Cell 2] 解压权重文件 (PDF 3.1 节 "下载 DeepSeek-R1 源码")

import zipfile

if not os.path.isdir(MODEL_DIR):
    print(f"解压 {MODEL_ZIP} ...")
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zf:
        zf.extractall(MODEL_DIR)
    print("解压完成")
else:
    print(f"已存在: {MODEL_DIR}")

# 列出解压内容确认
for f in sorted(os.listdir(MODEL_DIR)):
    print(f"  {f}")

# %% [Cell 3] 下载并配置 rknn-llm 仓库 (PDF 3.1 节)

import os

if not os.path.isdir("/content/rknn-llm-main"):
    !wget -q https://github.com/airockchip/rknn-llm/archive/refs/tags/release-v1.1.4.tar.gz
    !tar xzf release-v1.1.4.tar.gz
    !mv rknn-llm-release-v1.1.4 rknn-llm-main

REPO_DIR = "/content/rknn-llm-main"
EXPORT_DIR = os.path.join(
    REPO_DIR, "examples",
    "DeepSeek-R1-Distill-Qwen-1.5B_Demo", "export"
)
assert os.path.isdir(EXPORT_DIR), f"{EXPORT_DIR} 不存在"
print(f"EXPORT_DIR = {EXPORT_DIR}")

# %% [Cell 4] 生成量化校准数据 (PDF 3.1 节)
#    用 fp16 模型生成结果作为量化校准数据

os.chdir(EXPORT_DIR)

!python generate_data_quant.py -m {MODEL_DIR}

print("data_quant.json 已生成")

# %% [Cell 5] 模型转换 — W4A16 量化 (PDF 3.1 节)
#    W4A16 = 最小内存 (板子只有 3.7GB)

from rkllm.api import RKLLM

MODEL_OUT = "/content/DeepSeek-R1-Distill-Qwen-1.5B_W4A16_RK3588.rkllm"

llm = RKLLM()  # type: ignore[name-defined] # noqa: F821

# 加载 HuggingFace 权重
ret = llm.load_huggingface(model=MODEL_DIR, model_lora=None, device='cuda')
assert ret == 0, f"load_huggingface failed: {ret}"
print("load_huggingface OK")

# W4A16 量化 + 转换
ret = llm.build(
    do_quantization=True,
    optimization_level=1,
    quantized_dtype="W4A16",      # 最小内存模式
    quantized_algorithm="normal",
    target_platform="RK3588",
    num_npu_core=3,
    extra_qparams=None,
    dataset="./data_quant.json",
)
assert ret == 0, f"build failed: {ret}"
print("build OK")

# 导出
ret = llm.export_rkllm(MODEL_OUT)
assert ret == 0, f"export_rkllm failed: {ret}"

file_size_mb = os.path.getsize(MODEL_OUT) / (1024 * 1024)
print(f"转换完成: {MODEL_OUT} ({file_size_mb:.1f} MB)")

# %% [Cell 6] 下载模型到本地

from google.colab import files

files.download(MODEL_OUT)
print("下载已开始，保存到本地后 scp 到 RK3588 板子")

#!/usr/bin/env python3
"""
多模态情绪融合模块 — 基于置信度的自适应加权融合
"""
import numpy as np


def _entropy(probs):
    """归一化熵 → [0, 1]"""
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p)) / np.log(len(p))


def _confidence(probs):
    """置信度 = 1 - 归一化熵"""
    return 1.0 - _entropy(probs)


def fuse(eeg_probs=None, yolo_probs=None, eeg_weight=0.0, yolo_weight=0.0):
    """
    自适应加权融合 EEG 和 YOLO 情绪概率

    参数:
        eeg_probs:  [positive, neutral, negative]  or None
        yolo_probs: [positive, neutral, negative]  or None
        eeg_weight: 固定基础权重 (0=纯自适应), 不传则全自适应
        yolo_weight: 固定基础权重

    返回:
        np.array([positive, neutral, negative])  三分类概率 (sum=1)
    """
    eeg_p = np.asarray(eeg_probs, dtype=np.float64) if eeg_probs is not None else None
    yolo_p = np.asarray(yolo_probs, dtype=np.float64) if yolo_probs is not None else None

    # 双方都缺失 → 均匀分布
    if eeg_p is None and yolo_p is None:
        return np.full(3, 1.0 / 3.0, dtype=np.float64)

    # 单方缺失 → 回退
    if eeg_p is None:
        return yolo_p / yolo_p.sum()
    if yolo_p is None:
        return eeg_p / eeg_p.sum()

    # 双方都有 → 置信度加权
    c_eeg = _confidence(eeg_p)
    c_yolo = _confidence(yolo_p)

    w_eeg = eeg_weight + c_eeg * (1.0 - eeg_weight - yolo_weight)
    w_yolo = yolo_weight + c_yolo * (1.0 - eeg_weight - yolo_weight)

    total_w = w_eeg + w_yolo
    if total_w < 1e-9:
        return (eeg_p + yolo_p) / 2.0

    fused = (w_eeg * eeg_p + w_yolo * yolo_p) / total_w
    return fused / fused.sum()

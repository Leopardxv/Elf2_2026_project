#!/usr/bin/env python3
"""
ELF2 ↔ 服务器 二进制通信协议 编解码模块
协议版本 V1.0 - 严格遵循约束文档
"""

import struct
import zlib
import time
from typing import Tuple, Optional

# 帧结构常量
SYNC_MAGIC = 0xAA55          # 同步魔数
HEADER_SIZE = 17             # 固定帧头: 2+2+1+4+8 = 17 字节
CRC_SIZE = 4                 # CRC32 校验尾
FRAME_OVERHEAD = HEADER_SIZE + CRC_SIZE  # 帧开销 21 字节
MAX_PAYLOAD = 64 * 1024      # 最大载荷 64KB

# 消息类型
TYPE_HEARTBEAT = 0x00
TYPE_ATTENTION = 0x01
TYPE_ROBOT_VELOCITY = 0x02
TYPE_EMOTION = 0x03
TYPE_EEG = 0x04

MSG_TYPE_NAMES = {
    0x00: 'Heartbeat',
    0x01: 'Attention',
    0x02: 'RobotVelocity',
    0x03: 'Emotion',
    0x04: 'EEG',
}


def crc32(data: bytes) -> int:
    """计算 CRC32 (兼容标准 CRC-32/ISO-HDLC)"""
    return zlib.crc32(data) & 0xFFFFFFFF


def encode_frame(frame_id: int, msg_type: int, payload: bytes,
                 timestamp_ms: Optional[int] = None) -> bytes:
    """
    编码一帧二进制数据
    Args:
        frame_id: 帧序列号 0~65535
        msg_type: 消息类型 (0x00~0x04)
        payload: 载荷字节
        timestamp_ms: 毫秒时间戳, None 则自动取当前时间
    Returns:
        完整帧字节
    """
    if msg_type not in (0x00, 0x01, 0x02, 0x03, 0x04):
        raise ValueError(f'不支持的消息类型: 0x{msg_type:02X}')
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f'Payload 过大: {len(payload)} > {MAX_PAYLOAD}')

    if timestamp_ms is None:
        timestamp_ms = int(time.time() * 1000)

    header = struct.pack('>HHB I Q',
                          SYNC_MAGIC,
                          frame_id & 0xFFFF,
                          msg_type,
                          len(payload),
                          timestamp_ms)
    body = header + payload
    checksum = crc32(body)
    return body + struct.pack('>I', checksum)


def decode_frame(data: bytes) -> Optional[dict]:
    """
    解码一帧二进制数据
    Args:
        data: 完整帧字节 (从魔数到 CRC32)
    Returns:
        解码后的字典，或 None (解码失败)
    """
    if len(data) < FRAME_OVERHEAD:
        return None

    # 验证魔数
    sync = struct.unpack('>H', data[0:2])[0]
    if sync != SYNC_MAGIC:
        return None

    # 解析帧头
    sync_val, frame_id, msg_type, payload_len, timestamp_ms = \
        struct.unpack('>HHB I Q', data[:HEADER_SIZE])

    if sync_val != SYNC_MAGIC:
        return None
    if payload_len > MAX_PAYLOAD:
        return None
    if len(data) < HEADER_SIZE + payload_len + CRC_SIZE:
        return None

    # 提取载荷
    payload = data[HEADER_SIZE:HEADER_SIZE + payload_len]
    received_crc = struct.unpack('>I', data[HEADER_SIZE + payload_len:
                                             HEADER_SIZE + payload_len + CRC_SIZE])[0]

    # CRC32 校验: 帧头 + 载荷
    expected_crc = crc32(data[:HEADER_SIZE + payload_len])
    if received_crc != expected_crc:
        return None

    # 解码 payload
    result = {
        'frame_id': frame_id,
        'msg_type': msg_type,
        'msg_type_name': MSG_TYPE_NAMES.get(msg_type, f'Unknown(0x{msg_type:02X})'),
        'timestamp_ms': timestamp_ms,
        'payload_raw': payload,
    }

    decoded = _decode_payload(msg_type, payload)
    if decoded is not None:
        result['payload'] = decoded

    return result


def _decode_payload(msg_type: int, payload: bytes) -> Optional[dict]:
    """解码各类型 payload"""
    try:
        if msg_type == TYPE_ATTENTION:
            if len(payload) != 4:
                return None
            return {'attention': struct.unpack('>f', payload)[0]}

        elif msg_type == TYPE_ROBOT_VELOCITY:
            if len(payload) != 8:
                return None
            linear, angular = struct.unpack('>ff', payload)
            return {'linear_velocity': linear, 'angular_velocity': angular}

        elif msg_type == TYPE_EMOTION:
            if len(payload) != 12:
                return None
            pos, neu, neg = struct.unpack('>fff', payload)
            return {'positive': pos, 'neutral': neu, 'negative': neg}

        elif msg_type == TYPE_EEG:
            if len(payload) < 3:
                return None
            channels = payload[0]
            samples = struct.unpack('>H', payload[1:3])[0]
            expected_data_size = channels * samples * 4
            if len(payload) != 3 + expected_data_size:
                return None
            # 解析 int32 数组
            eeg_flat = struct.unpack(f'>{channels * samples}i',
                                      payload[3:3 + expected_data_size])
            # 重塑为二维: (samples, channels) — 按"先跨通道再跨时间"
            import array
            eeg = []
            idx = 0
            for s in range(samples):
                row = [eeg_flat[s * channels + c] for c in range(channels)]
                eeg.append(row)
            return {'channels': channels, 'samples': samples, 'eeg_data': eeg}

        elif msg_type == TYPE_HEARTBEAT:
            return {'heartbeat': True}

        return None
    except struct.error:
        return None


# ---- 便捷编码函数 ----

def encode_attention(frame_id: int, score: float,
                     timestamp_ms: Optional[int] = None) -> bytes:
    """编码注意力帧 (0x01)"""
    payload = struct.pack('>f', score)
    return encode_frame(frame_id, TYPE_ATTENTION, payload, timestamp_ms)


def encode_robot_velocity(frame_id: int, linear: float, angular: float,
                          timestamp_ms: Optional[int] = None) -> bytes:
    """编码小车速度帧 (0x02)"""
    payload = struct.pack('>ff', linear, angular)
    return encode_frame(frame_id, TYPE_ROBOT_VELOCITY, payload, timestamp_ms)


def encode_emotion(frame_id: int, positive: float, neutral: float,
                   negative: float,
                   timestamp_ms: Optional[int] = None) -> bytes:
    """编码情绪帧 (0x03)"""
    payload = struct.pack('>fff', positive, neutral, negative)
    return encode_frame(frame_id, TYPE_EMOTION, payload, timestamp_ms)


def encode_eeg(frame_id: int, channels: int, samples: int,
               data: list,  # list of list: (samples, channels), e.g. [[ch1,ch2,...], ...]
               timestamp_ms: Optional[int] = None) -> bytes:
    """
    编码脑电帧 (0x04)
    Args:
        channels: 通道数
        samples: 采样点数
        data: 采样数据 — 按"先跨通道再跨时间"排列
              外层是采样点 [sample0, sample1, ...]
              内层是通道值 [ch1_val, ch2_val, ...]
    """
    payload_head = struct.pack('>BH', channels, samples)
    # 打包为 int32 (ADC 原始值 → 乘以 1000 保留精度)
    flat = []
    for s in range(samples):
        for c in range(channels):
            val = data[s][c]
            flat.append(int(val))
    payload_data = struct.pack(f'>{channels * samples}i', *flat)
    return encode_frame(frame_id, TYPE_EEG, payload_head + payload_data, timestamp_ms)


def encode_heartbeat(frame_id: int,
                     timestamp_ms: Optional[int] = None) -> bytes:
    """编码心跳帧 (0x00)"""
    return encode_frame(frame_id, TYPE_HEARTBEAT, b'', timestamp_ms)

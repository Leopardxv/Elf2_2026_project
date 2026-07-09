# 小车通信系统 (xiaoche_communication)

RK3588 与 A733 双板 ROS 2 双向通信。

## 消息定义 (FiveFloats)

| 字段 | 类型 | 说明 |
|------|------|------|
| `attention` | float32 | **注意力系数** |
| `target_speed` | float32 | 目标速度 |
| `current_speed` | float32 | 当前实际速度 |
| `data4` | float32 | 预留 |
| `data5` | float32 | 预留 |

### 注意力系数 (attention) 语义

```
  1.0 ─── 完全放松、不专注
  ...
  0.5 ─── 中等
  ...
  0.0 ─── 极度紧张、高度专注
```

- **越接近 1.0**：用户状态越放松，注意力分散
- **越接近 0.0**：用户状态越紧张，注意力高度集中

### 注意

不要将其误解为"越接近 1 越专注"，该系数是一个**反向指标**：
高值 = 放松，低值 = 专注。

## 话题 (Topics)

| 话题 | 方向 |
|------|------|
| `/topic_rk3588_to_a733` | RK3588 → A733 |
| `/topic_a733_to_rk3588` | A733 → RK3588 |

## 构建与运行

```bash
# 构建
source /opt/ros/humble/setup.bash
export COLCON_PYTHON_EXECUTABLE=/usr/bin/python3
cd ~/Projects/xiaoche_ws && colcon build

# 运行
source install/setup.bash
ros2 run xiaoche_communication node_rk3588   # 终端1
ros2 run xiaoche_communication node_a733     # 终端2
```

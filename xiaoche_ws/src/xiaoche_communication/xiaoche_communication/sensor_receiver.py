#!/usr/bin/env python3
"""
RK3588 传感器接收器 — 从小车 A733 接收点云和气体浓度数据。

维护最新传感器状态，供 voice_assistant 等模块查询。
"""
import rclpy
from rclpy.node import Node
from custom_msgs.msg import GasSensor, PointCloud
import threading
import json


class SensorState:
    """线程安全的最新传感器数据"""
    def __init__(self):
        self._lock = threading.Lock()
        self._gas = {}      # {gas_type: {"concentration": float, "temperature": float}}
        self._pointcloud = {"count": 0, "points": []}
        self._updated = False

    def set_gas(self, gas_type: str, concentration: float, temperature: float):
        with self._lock:
            self._gas[gas_type] = {"concentration": concentration, "temperature": temperature}
            self._updated = True

    def set_pointcloud(self, count: int, x: list, y: list, z: list):
        with self._lock:
            self._pointcloud = {"count": count, "points": list(zip(x, y, z))}
            self._updated = True

    def get_gas(self) -> dict:
        with self._lock:
            return dict(self._gas)

    def get_pointcloud(self) -> dict:
        with self._lock:
            return dict(self._pointcloud)

    def get_gas_summary(self) -> str:
        """返回气体浓度的文字摘要"""
        gas = self.get_gas()
        if not gas:
            return "无气体数据"
        parts = []
        for name, data in gas.items():
            conc = data["concentration"]
            temp = data["temperature"]
            if name == "CH4":
                level = "安全" if conc < 0.5 else ("警戒" if conc < 1.0 else "危险！需立即撤离")
                parts.append(f"甲烷(CH4) {conc:.1f}% ({level})")
            elif name == "CO":
                level = "安全" if conc < 24 else ("注意" if conc < 50 else "危险！一氧化碳超标")
                parts.append(f"一氧化碳(CO) {conc:.0f}ppm ({level})")
            elif name == "H2S":
                level = "安全" if conc < 6.6 else ("注意" if conc < 10 else "危险！硫化氢超标")
                parts.append(f"硫化氢(H2S) {conc:.1f}ppm ({level})")
            elif name == "O2":
                level = "正常" if 19.5 <= conc <= 23.5 else ("偏低" if conc < 19.5 else "偏高")
                parts.append(f"氧气(O2) {conc:.1f}% ({level})")
            else:
                parts.append(f"{name} {conc:.2f}ppm")
        return "；".join(parts)

    def get_obstacle_summary(self) -> str:
        """返回障碍物/点云摘要"""
        pc = self.get_pointcloud()
        if pc["count"] == 0:
            return "未检测到障碍物"
        points = pc["points"]
        if not points:
            return "无点云数据"
        # 统计最近障碍物距离和方向
        distances = [(p[0]**2 + p[1]**2 + p[2]**2)**0.5 for p in points]
        min_dist = min(distances)
        if min_dist < 0.5:
            return f"⚠️ 前方 {min_dist:.1f}m 有障碍物，立即停车！"
        elif min_dist < 1.5:
            return f"前方 {min_dist:.1f}m 检测到障碍物，请减速慢行"
        return f"前方 {min_dist:.1f}m 外有物体，环境安全"


class SensorReceiver(Node):
    """订阅 A733 传感器话题，更新共享状态"""

    def __init__(self, state: SensorState = None):
        super().__init__("sensor_receiver_rk3588")
        self._state = state or SensorState()

        self._gas_sub = self.create_subscription(
            GasSensor, "/topic_a733_gas", self._gas_callback, 10)
        self._pc_sub = self.create_subscription(
            PointCloud, "/topic_a733_pointcloud", self._pc_callback, 10)

        self.get_logger().info("Sensor receiver ready — listening for gas + pointcloud")

    def _gas_callback(self, msg: GasSensor):
        self._state.set_gas(msg.gas_type, msg.concentration, msg.temperature)
        self.get_logger().debug(
            f"Gas: {msg.gas_type}={msg.concentration} @ {msg.temperature}°C")

    def _pc_callback(self, msg: PointCloud):
        self._state.set_pointcloud(msg.count, list(msg.x), list(msg.y), list(msg.z))
        self.get_logger().debug(f"PointCloud: {msg.count} points")


# ================================================================
# 全局单例，供其他模块（voice_assistant, bci_dashboard）读取
# ================================================================
_sensor_state: SensorState = None


def get_sensor_state() -> SensorState:
    global _sensor_state
    if _sensor_state is None:
        _sensor_state = SensorState()
    return _sensor_state


def start_receiver():
    """启动传感器接收器（阻塞，在独立线程中调用）"""
    rclpy.init(args=[])
    node = SensorReceiver(state=get_sensor_state())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    start_receiver()


if __name__ == "__main__":
    main()

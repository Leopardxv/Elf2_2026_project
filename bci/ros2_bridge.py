#!/usr/bin/env python3
"""
ROS2 桥接模块 — 在 PyQt5 仪表盘内运行 rclpy
发布: /topic_rk3588_to_a733 (FiveFloats)  — 注意力
订阅: /topic_a733_to_rk3588 (FiveFloats)  — 小车速度
"""
import threading
import rclpy
from rclpy.node import Node
from custom_msgs.msg import FiveFloats
from PyQt5 import QtCore


class ROS2Bridge(QtCore.QObject):
    """ROS2 桥接器，使用 QTimer + spin_once 在 GUI 线程驱动"""

    velocity_updated = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._node = None
        self._pub = None
        self._latest_attention = 1.0
        self._lock = threading.Lock()
        self._init_ros()

    def _init_ros(self):
        if not rclpy.ok():
            rclpy.init(args=[])
        self._node = rclpy.create_node('bci_dashboard_node')
        self._pub = self._node.create_publisher(
            FiveFloats, '/topic_rk3588_to_a733', 10)

        self._node.create_subscription(
            FiveFloats, '/topic_a733_to_rk3588',
            self._on_velocity_msg, 10)

        self._spin_timer = QtCore.QTimer(self)
        self._spin_timer.timeout.connect(self._spin_once)
        self._spin_timer.start(50)

    def _spin_once(self):
        if rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0)

    def _on_velocity_msg(self, msg):
        self.velocity_updated.emit(msg.target_speed, msg.data4)

    def publish_attention(self, value):
        msg = FiveFloats()
        msg.attention = float(value)
        msg.target_speed = 0.0
        msg.current_speed = 0.0
        msg.data4 = 0.0
        msg.data5 = 0.0
        if self._pub is not None:
            self._pub.publish(msg)

    def shutdown(self):
        self._spin_timer.stop()
        if self._node is not None:
            self._node.destroy_node()

#!/usr/bin/env python3
# 程序主要结构：
# 1. 导入 ROS 2 核心库与 custom_msgs 自定义消息类型。
# 2. 初始化 A733Node 类，创建向 RK3588 发送的发布者，和接收 RK3588 数据的订阅者。
# 3. 设置定时器，每 1 秒打包发送包含 attention, target_speed 等 5 个具体特征数据。
# 4. 编写回调函数，实时解析并打印收到的 RK3588 具名数据。
#
# attention 注意力系数语义：
#   ≈ 1.0 → 放松/不专注  |  ≈ 0.0 → 紧张/高度专注

import rclpy
from rclpy.node import Node
from custom_msgs.msg import FiveFloats

class A733Node(Node):
    def __init__(self):
        super().__init__('xiaoche_communication_a733')
        # 发布给 RK3588 的话题
        self.publisher_ = self.create_publisher(FiveFloats, '/topic_a733_to_rk3588', 10)
        # 订阅来自 RK3588 的话题
        self.subscription = self.create_subscription(FiveFloats, '/topic_rk3588_to_a733', self.listener_callback, 10)
        
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('通信节点 A733 已启动，正在收发数据...')

    def timer_callback(self):
        msg = FiveFloats()
        # 填充属于 A733 的具名特征数据
        msg.attention = 0.8        # 注意力系数: ≈1.0 放松, ≈0.0 高度专注
        msg.target_speed = 2.0     # 目标速度
        msg.current_speed = 1.8    # 当前实际速度
        msg.data4 = 9.4            # 预留数据
        msg.data5 = 9.5            # 预留数据
        
        self.publisher_.publish(msg)
        self.get_logger().info(
            f'发送 A733 -> RK3588: [Att: {msg.attention:.2f}, '
            f'TarSpd: {msg.target_speed:.2f}, CurSpd: {msg.current_speed:.2f}, '
            f'D4: {msg.data4:.2f}, D5: {msg.data5:.2f}]'
        )

    def listener_callback(self, msg):
        self.get_logger().info(
            f'收到 RK3588 -> A733: [Att: {msg.attention:.2f}, '
            f'TarSpd: {msg.target_speed:.2f}, CurSpd: {msg.current_speed:.2f}, '
            f'D4: {msg.data4:.2f}, D5: {msg.data5:.2f}]'
        )

def main(args=None):
    rclpy.init(args=args)
    node = A733Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
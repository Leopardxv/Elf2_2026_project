import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

class RealTimePlot16:
    def __init__(self, board_shim):
        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()
        
        # 获取全部 16 个 EEG 通道索引
        self.ex_channels = BoardShim.get_eeg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.window_size = 4  # 显示最近 4 秒
        self.num_points = self.window_size * self.sampling_rate

        # 初始化绘图窗口
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(title='OpenBCI 16-Channel Monitor (Linux)', size=(1000, 900))
        self.win.show()

        self.plots = []
        self.curves = []
        
        # 16 通道纵向排列，分两列显示（每列 8 个）以防窗口太长
        for i in range(len(self.ex_channels)):
            # row = i % 8, col = 0 if i < 8 else 1
            p = self.win.addPlot(row=i % 8, col=i // 8)
            p.showAxis('left', True)
            p.setMenuEnabled(False)
            p.setClipToView(True)
            p.setLabel('left', f'CH {i+1}', units='uV')
            
            # 使用不同的颜色区分
            color = (100, 200, 255) if i < 8 else (255, 150, 100)
            curve = p.plot(pen=pg.mkPen(color=color, width=1))
            
            self.plots.append(p)
            self.curves.append(curve)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50) 
        
        sys.exit(self.app.exec_())

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        if data.shape[1] == 0:
            return

        for i, channel in enumerate(self.ex_channels):
            chan_data = data[channel]
            if len(chan_data) > 0:
                # 简单滤波：去直流偏置
                chan_data = chan_data - np.mean(chan_data)
                self.curves[i].setData(chan_data)

def main():
    BoardShim.enable_dev_board_logger()
    
    params = BrainFlowInputParams()
    # Linux 常见的串口路径是 /dev/ttyUSB0 或 /dev/ttyACM0
    # 你可以通过执行 'ls /dev/ttyUSB*' 来确认
    params.serial_port = "/dev/ttyUSB0" 
    
    # 核心：设置为 16 通道板 ID (2)
    board_id = BoardIds.CYTON_DAISY_BOARD

    board = None
    try:
        board = BoardShim(board_id, params)
        board.prepare_session()
        board.start_stream(45000)
        print("16 通道启动成功，正在绘图...")
        RealTimePlot16(board)
    except Exception as e:
        print(f"启动失败: {e}")
    finally:
        if board and board.is_prepared():
            board.stop_stream()
            board.release_session()

if __name__ == "__main__":
    main()

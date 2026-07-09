"""
程序主要结构：
1. 导入库：numpy 用于矩阵操作，scipy.io 用于读取 .mat 文件，os 用于路径检查。
2. 路径设置：指定输入的 seed_save 目录和输出的 data_cv5fold 目录。
3. 数据切片循环 (仅针对测试的 1 个受试者)：
   - 遍历 3 个 Session。
   - 读取每个 Session 下的 15 个 Trial 的 .mat 文件。
   - 以 200 个采样点（1秒）为固定窗口滑动，将连续的脑电数据切分成多个 (62, 200) 的数据块。
   - 提取对应的标签，并与数据块对齐。
4. 存储张量：将组合好的特征张量和标签张量保存为 .npy 格式，供 PyTorch 直接拉取。
"""
import numpy as np
import scipy.io
import os

# 设置输入和输出路径（已替换为你电脑上的实际路径）
root_path = 'D:/EEGdata/SEED/seed_save/'
save_path = 'D:/EEGdata/SEED/data_cv5fold/'

# 如果输出文件夹不存在，自动创建
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 因为我们刚才只处理了 Subject 1，所以这里循环范围改成 1 (即 i=0，代表受试者1)
for i in range(1):
    for j in range(3): # 遍历 3 个 Session
        one_session = []
        one_session_label = []
        
        # 遍历 15 个 Trial
        for k in range(15):
            one_trial = []
            
            # 读取上一步 MATLAB 生成的单 Trial 文件
            mat_file_path = os.path.join(root_path, 'S%d_%d_%d.mat' % (i+1, j+1, k+1))
            trial_tmp = scipy.io.loadmat(mat_file_path)
            trial_data = trial_tmp['trial_data']
            trial_label = np.squeeze(trial_tmp['trial_label'])
            
            # 按照 200 个采样点（1秒）为步长进行切片
            trial_number = np.int32(trial_data.shape[1] / 200)
            for tmp_num in range(trial_number):
                one_trial.append(trial_data[:, tmp_num*200 : (tmp_num+1)*200])
                
            one_trial_label = [trial_label] * trial_number
            
            one_session.append(one_trial)
            one_session_label.append(one_trial_label)

        # 转换为 NumPy 数组并允许 object 格式（因为每个电影的时长不同，切片数量不一致）
        data = np.array(one_session, dtype=object)
        label = np.array(one_session_label, dtype=object)
        
        # 保存为 .npy 供后续 PyTorch 的 DataLoader 读取
        np.save(os.path.join(save_path, 'S%d_session%d.npy' % (i+1, j+1)), data)
        np.save(os.path.join(save_path, 'S%d_session%d_label.npy' % (i+1, j+1)), label)

        print('Finished Subject %d Session %d' % (i+1, j+1))

print("测试数据张量切片与保存全部完成！")
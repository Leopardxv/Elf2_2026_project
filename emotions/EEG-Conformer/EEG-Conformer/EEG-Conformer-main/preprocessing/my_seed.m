% =========================================================================
% 程序主要结构：
% 1. 环境与路径配置：设置预处理输入目录、输出保存目录，并指定要测试的 3 个 Session 文件。
% 2. 滤波器与标签初始化：
%    - 设计 4-47 Hz Chebyshev II 零相位带通滤波器。
%    - 载入 SEED 官方固定的 15 段电影情感标签序列。
% 3. 数据处理主循环：
%    - 遍历 3 个 .mat 文件加载数据。
%    - 动态遍历提取 15 个 Trial (应对变量名为 eeg1, eeg_1 或 xxx_eeg1 的情况)。
%    - 严格执行 Z-score 均值方差标准化。
%    - 严格执行 4-47 Hz 带通滤波。
% 4. 拆分保存：将处理完的 15 个 Trial 分别打包为独立的 .mat 文件输出。
% =========================================================================

% --- 1. 环境与路径配置 ---
input_dir = 'D:/EEGdata/SEED/SEED/Preprocessed_EEG/';
output_dir = 'D:/EEGdata/SEED/seed_save/';

% 如果输出目录不存在则自动创建
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 手动指定要测试的 3 个 .mat 文件名（不加后缀）
test_files = {'1_20131027', '1_20131030', '1_20131107'};

% --- 2. 滤波器与标签初始化 ---
% SEED 数据集标准的 15 组情感标签 (1:积极, 0:中性, -1:消极)
seed_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1];

% 严格还原原程序的带通滤波器 (4-47 Hz)
fc = 200; % 采样率 200 Hz
Wl = 4; Wh = 47; 
Wn = [Wl*2 Wh*2]/fc;
[b,a] = cheby2(6, 60, Wn);

% --- 3. 数据处理主循环 ---
% j 代表 Session (1 到 3)
for j = 1:length(test_files)
    file_name = test_files{j};
    file_path = fullfile(input_dir, [file_name, '.mat']);
    
    fprintf('正在处理测试文件: %s.mat\n', file_name);
    session_data = load(file_path);
    fields = fieldnames(session_data); % 获取结构体里的所有变量名
    
    % k 代表 Trial (1 到 15)
    for k = 1:15 
        % 动态匹配变量名，防止有的文件叫 eeg1，有的叫 eeg_1 甚至 djc_eeg1
        target_suffix = sprintf('eeg%d', k);
        target_suffix_alt = sprintf('eeg_%d', k);
        data_field = '';
        
        for f = 1:length(fields)
            if endsWith(fields{f}, target_suffix, 'IgnoreCase', true) || ...
               endsWith(fields{f}, target_suffix_alt, 'IgnoreCase', true)
                data_field = fields{f};
                break;
            end
        end
        
        if isempty(data_field)
            error('在文件 %s 中找不到对应 Trial %d 的 eeg 变量！', file_name, k);
        end
        
        % 提取出 (62, time_points) 的数据矩阵
        trial_data = session_data.(data_field);
        
        % 严格按照原始代码顺序：先求均值标准差进行 Z-score，再滤波
        trial_mean = mean(trial_data, 2);
        trial_std = std(trial_data, 1, 2); 
        trial_data = (trial_data - trial_mean) ./ trial_std;
        trial_data = filtfilt(b, a, trial_data);
        
        % 获取对应的标签
        trial_label = seed_labels(k);
        
        % --- 4. 拆分保存 ---
        % 文件命名严格遵循原格式：S受试者号_Session号_Trial号.mat
        % 因为我们测的是被试1的三个文件，所以受试者号固定填1，Session号是j
        saveDir = fullfile(output_dir, sprintf('S1_%d_%d.mat', j, k));
        save(saveDir, 'trial_data', 'trial_label');
    end
end

fprintf('\n==== 3 个 Session 的提取与预处理测试已全部完成！ ====\n');
fprintf('请检查文件夹: %s\n', output_dir);
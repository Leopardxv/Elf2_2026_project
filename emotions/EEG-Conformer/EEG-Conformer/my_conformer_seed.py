"""
程序主要结构：
1. 环境配置与依赖导入：设定单张本地 GPU (gpus=[0])，并引入 PyTorch、Matplotlib 等相关依赖。
2. 数据增强方法 (augment_snr)：
   - 实现论文中提出的 S&R (Segmentation and Reconstruction) 时域数据增强策略。
   - 将同一 Batch 内同类别的 EEG 数据在时间维度等分为 N_s=8 份，在维持时间顺序的前提下随机拼接。
3. 网络结构定义 (ViT/Conformer)：
   - PatchEmbedding: 使用针对 SEED 数据集的 62 通道卷积核 (62, 1) 提取空间特征，以及时间维度的平均池化。
   - MultiHeadAttention & TransformerEncoder: 自注意力机制提取全局时序特征 (头数修改为对齐论文的 10 头)。
   - ClassificationHead: 将 280 维特征映射到 3 分类。
4. 训练控制类 (ExGAN):
   - 初始化参数: 学习率、批次大小、训练轮数等。
   - get_source_data: 加载对应 Subject 数据，执行 5-fold 划分与 Z-score 标准化。
   - train: 核心训练循环。
     > 加入了 augment_snr 数据增强。
     > 加入了记录 epoch_loss, epoch_train_acc, epoch_test_acc 的列表。
     > 训练结束后，利用 Matplotlib 绘制双 Y 轴的 Loss & Accuracy 曲线，并保存到本地。
5. 主函数 (main):
   - 自动创建日志和结果文件夹，针对 Subject 1 执行 5 折交叉验证并汇总结果。
"""

import os
gpus = [0] 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import random
import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch.backends import cudnn
import matplotlib.pyplot as plt  # 新增用于绘制训练动态图

# 固定随机种子以保证结果可复现
cudnn.benchmark = False
cudnn.deterministic = True


# --- 2. 数据增强模块 (S&R) ---
def augment_snr(x, y, n_segments=8):
    """
    Segmentation and Reconstruction (S&R) 数据增强
    x shape: [Batch_size, 1, Channels, TimeSamples] (例如: [B, 1, 62, 200])
    y shape: [Batch_size]
    """
    aug_x = torch.zeros_like(x)
    unique_classes = torch.unique(y)
    
    for c in unique_classes:
        # 获取当前 batch 中属于类别 c 的所有样本的索引
        idx = (y == c).nonzero(as_tuple=True)[0]
        num_samples = len(idx)
        
        # 如果该类别在当前 batch 中没有样本，直接跳过
        if num_samples == 0:
            continue
            
        class_x = x[idx] # [K, 1, C, T]
        T = class_x.shape[-1]
        segment_len = T // n_segments
        
        aug_class_x = torch.zeros_like(class_x)
        # 对每一个分段进行重构
        for i in range(n_segments):
            start = i * segment_len
            end = (i + 1) * segment_len if i < n_segments - 1 else T
            
            # 从该类别的样本池中，为每一个新样本随机抽取当前段
            rand_idx = torch.randint(0, num_samples, (num_samples,), device=x.device)
            aug_class_x[:, :, :, start:end] = class_x[rand_idx, :, :, start:end]
            
        aug_x[idx] = aug_class_x
        
    return aug_x


# --- 3. 网络结构定义 ---
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (62, 1), (1, 1)), # 针对 SEED 的 62 个电极
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    # 修改：按照论文将默认自注意力头数调整为 10
    def __init__(self, emb_size, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(280, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes) 
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out

class ViT(nn.Sequential):
    # 修改：按照论文将深度 depth 调整为 6
    def __init__(self, emb_size=40, depth=6, n_classes=3, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

# --- 4. 训练控制类 ---
class ExGAN():
    def __init__(self, nsub, fold):
        super(ExGAN, self).__init__()
        self.batch_size = 200
        self.n_epochs = 400  # 可以根据实际收敛情况调整
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.root = 'D:/EEGdata/SEED/data_cv5fold/'

        # 创建日志和图片保存路径
        self.log_dir = "./results/seed/5-fold/real/"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_write = open(os.path.join(self.log_dir, f"log_subject{self.nSub}_fold{fold+1}.txt"), "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = ViT(n_classes=3).cuda()
        self.model = nn.DataParallel(self.model, device_ids=[0])

    def get_source_data(self, fold):
        self.all_data = np.load(os.path.join(self.root, f'S{self.nSub}_session1.npy'), allow_pickle=True)
        self.all_label = np.load(os.path.join(self.root, f'S{self.nSub}_session1_label.npy'), allow_pickle=True)
        
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []

        for tri in range(np.shape(self.all_data)[0]):
            tmp_tri = np.array(self.all_data[tri])
            tmp_tri_label = np.array(self.all_label[tri])

            one_fold_num = np.shape(tmp_tri)[0] // 5
            tri_num =  one_fold_num * 5
            tmp_tri_idx = np.arange(tri_num)
            test_idx = np.arange(one_fold_num * fold, one_fold_num * (fold+1))
            train_idx = np.delete(tmp_tri_idx, test_idx)

            self.train_data.append(tmp_tri[train_idx])
            self.train_label.append(tmp_tri_label[train_idx])
            self.test_data.append(tmp_tri[test_idx])
            self.test_label.append(tmp_tri_label[test_idx])
        
        self.train_data = np.concatenate(self.train_data)
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.concatenate(self.train_label)
        self.test_data = np.concatenate(self.test_data)
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.concatenate(self.test_label)

        shuffle_num = np.random.permutation(len(self.train_data))
        self.train_data = self.train_data[shuffle_num, :, :, :]
        self.train_label = self.train_label[shuffle_num]

        target_mean = np.mean(self.train_data)
        target_std = np.std(self.train_data)
        self.train_data = (self.train_data - target_mean) / target_std
        self.test_data = (self.test_data - target_mean) / target_std

        return self.train_data, self.train_label, self.test_data, self.test_label

    def train(self, fold):
        img, label, test_data, test_label = self.get_source_data(fold)

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label + 1).long()

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data).float()
        test_label = torch.from_numpy(test_label + 1).long()
        test_data = Variable(test_data.cuda())
        test_label = Variable(test_label.cuda())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        bestAcc = 0
        averAcc = 0
        num = 0
        
        # --- 用于绘制曲线的记录列表 ---
        epoch_losses = []
        epoch_train_accs = []
        epoch_test_accs = []

        print(f"--- 正在开始 Fold {fold+1} 的训练 ---")
        for e in range(self.n_epochs):
            self.model.train()
            batch_loss_sum = 0
            batch_train_acc_sum = 0
            batch_count = 0
            
            for i, (batch_img, batch_label) in enumerate(self.dataloader):
                batch_img = Variable(batch_img.cuda())
                batch_label = Variable(batch_label.cuda())

                # === S&R 数据增强 ===
                # 在送入模型前，对输入进行分割和重构增强 (切分为 8 段)
                batch_img = augment_snr(batch_img, batch_label, n_segments=8)

                tok, outputs = self.model(batch_img)
                loss = self.criterion_cls(outputs, batch_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 累加用于计算 Epoch 维度的平均训练数据
                batch_loss_sum += loss.item()
                train_pred = torch.max(outputs, 1)[1]
                batch_train_acc_sum += float((train_pred == batch_label).cpu().numpy().astype(int).sum()) / float(batch_label.size(0))
                batch_count += 1
            
            avg_train_loss = batch_loss_sum / batch_count
            avg_train_acc = batch_train_acc_sum / batch_count

            # 测试过程
            self.model.eval()
            with torch.no_grad():
                Tok, Cls = self.model(test_data)
                y_pred = torch.max(Cls, 1)[1]
                test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                
                print('Epoch: %d  Train loss: %.4f  Train acc: %.4f  Test acc: %.4f' % 
                      (e, avg_train_loss, avg_train_acc, test_acc))
                
                # 记录指标以供绘图
                epoch_losses.append(avg_train_loss)
                epoch_train_accs.append(avg_train_acc)
                epoch_test_accs.append(test_acc)
                
                self.log_write.write(str(e) + "    " + str(test_acc) + "\n")
                num += 1
                averAcc += test_acc
                if test_acc > bestAcc:
                    bestAcc = test_acc

        averAcc = averAcc / num
        print('The average accuracy of fold %d is: %.4f' % (fold+1, averAcc))
        print('The best accuracy of fold %d is: %.4f' % (fold+1, bestAcc))
        self.log_write.write('The average accuracy of fold%d is: ' %(fold+1) + str(averAcc) + "\n")
        self.log_write.write('The best accuracy fold%d is: ' %(fold+1) + str(bestAcc) + "\n")
        
        # --- 训练结束：绘制双 Y 轴曲线图 ---
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss', color=color)
        ax1.plot(range(self.n_epochs), epoch_losses, color=color, label='Training Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        # 实例化第二个共享 X 轴的坐标轴
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Accuracy', color=color)  
        ax2.plot(range(self.n_epochs), epoch_train_accs, color='tab:orange', linestyle='--', label='Training Accuracy')
        ax2.plot(range(self.n_epochs), epoch_test_accs, color=color, label='Test Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # 确保图例和标签不会超出边界
        plt.title(f'Subject {self.nSub} Fold {fold+1} Training Process')
        
        # 合并图例
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

        # 保存图片到指定路径
        plot_path = os.path.join(self.log_dir, f"sub{self.nSub}_fold{fold+1}_curve.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"训练曲线已保存至: {plot_path}")
        
        return bestAcc, averAcc

# --- 5. 主函数 ---
def main():
    best = 0
    aver = 0
    
    res_dir = "./results/seed/5-fold/"
    os.makedirs(res_dir, exist_ok=True)
    result_write = open(os.path.join(res_dir, "sub_result.txt"), "w")

    for i in range(1): # 目前针对单个被试测试
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(2021)

        result_write.write('--------------------------------------------------\n')
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        
        print(f'\n========== Subject {i+1} ==========')
        result_write.write(f'Subject {i + 1} : Seed is: {seed_n}\n')

        bestAcc = 0
        averAcc = 0

        for fold in range(5):
            exgan = ExGAN(i + 1, fold)
            ba, aa = exgan.train(fold)
            
            result_write.write('Best acc of fold ' + str(fold+1) + ' is: ' + str(ba) + "\n")
            result_write.write('Aver acc of fold ' + str(fold+1) + ' is: ' + str(aa) + "\n")
            bestAcc += ba
            averAcc += aa

        bestAcc /= 5
        averAcc /= 5
        result_write.write('5-fold Best acc is: ' + str(bestAcc) + "\n")
        result_write.write('5-fold Aver acc is: ' + str(averAcc) + "\n")
        
        best += bestAcc
        aver += averAcc
        endtime = datetime.datetime.now()
        print('Subject %d duration: ' % (i+1) + str(endtime - starttime))

    best /= 1
    aver /= 1

    result_write.write('--------------------------------------------------\n')
    result_write.write('All subject Best accuracy is: ' + str(best) + "\n")
    result_write.write('All subject Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()

if __name__ == "__main__":
    print("开始训练时间:", time.asctime(time.localtime(time.time())))
    main()
    print("结束训练时间:", time.asctime(time.localtime(time.time())))
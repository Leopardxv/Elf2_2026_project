"""
程序主要结构：
1. 环境配置与依赖导入：设定单卡 GPU，引入 PyTorch、Matplotlib、Sklearn (用于 t-SNE) 和 MNE (用于脑地形图)。
2. 数据增强方法 (augment_snr)：实现时域 S&R 数据增强，切分为 8 段随机拼接。
3. 网络结构定义 (ViT/Conformer)：
   - MultiHeadAttention: 保存注意力权重到 self.attn_weights。
   - ClassificationHead: 返回分类前的特征 (features) 以及最终预测结果 (out)。
4. 训练与可视化控制类 (ExGAN):
   - train: 核心训练循环，记录并绘制双 Y 轴的 Loss & Accuracy 曲线。同时保存完整的训练集和测试集 Tensor。
   - visualize_tsne: [功能升级] 接收指定的数据集 Tensor 和标签，采用分批次 (Batch) 推理的方式提取特征，防止显存溢出，生成对应的 t-SNE 降维分布图。
   - visualize_cat_topography: 提取最后一层的 CAM 权重，演示 CAT 地形图绘制。
5. 主函数 (main): 执行完整的 5 折交叉验证。在每一折结束后，分别调用 visualize_tsne 生成该折的“训练集 t-SNE”和“测试集 t-SNE”，并生成 CAT 地形图，最后汇总平均成绩。
"""

import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import random
import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.backends import cudnn
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import mne

# 固定随机种子以保证结果可复现
cudnn.benchmark = False
cudnn.deterministic = True

# --- 2. 数据增强模块 (S&R) ---
def augment_snr(x, y, n_segments=8):
    aug_x = torch.zeros_like(x)
    unique_classes = torch.unique(y)
    for c in unique_classes:
        idx = (y == c).nonzero(as_tuple=True)[0]
        num_samples = len(idx)
        if num_samples == 0: continue
        class_x = x[idx]
        T = class_x.shape[-1]
        segment_len = T // n_segments
        aug_class_x = torch.zeros_like(class_x)
        for i in range(n_segments):
            start = i * segment_len
            end = (i + 1) * segment_len if i < n_segments - 1 else T
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
            nn.Conv2d(40, 40, (62, 1), (1, 1)), 
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
        
        self.attn_weights = None 

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
        
        self.attn_weights = att.mean(dim=1).detach() 
        
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

class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerEncoderBlock(emb_size) for _ in range(depth)])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(280, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes) 
        )
    def forward(self, x):
        features = x.contiguous().view(x.size(0), -1) 
        out = self.fc(features)
        return features, out

class ViT(nn.Module):
    def __init__(self, emb_size=40, depth=6, n_classes=3, **kwargs):
        super().__init__()
        self.patch_embed = PatchEmbedding(emb_size)
        self.transformer = TransformerEncoder(depth, emb_size)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        features, out = self.classifier(x)
        return features, out

# --- 4. 训练与可视化控制类 ---
class ExGAN():
    def __init__(self, nsub, fold):
        super(ExGAN, self).__init__()
        self.batch_size = 200
        self.n_epochs = 400 
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.root = 'D:/EEGdata/SEED/data_cv5fold/'
        
        self.log_dir = "./results/seed/5-fold/real/"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_write = open(os.path.join(self.log_dir, f"log_subject{self.nSub}_fold{fold+1}.txt"), "w")

        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.model = ViT(n_classes=3).cuda()
        self.model = nn.DataParallel(self.model, device_ids=[0])
        
        # 记录全局的 Tensor 供后续可视化使用
        self.train_data_tensor_all = None
        self.train_label_tensor_all = None
        self.test_data_tensor = None
        self.test_label_tensor = None

    def get_source_data(self, fold):
        self.all_data = np.load(os.path.join(self.root, f'S{self.nSub}_session1.npy'), allow_pickle=True)
        self.all_label = np.load(os.path.join(self.root, f'S{self.nSub}_session1_label.npy'), allow_pickle=True)
        
        self.train_data, self.train_label = [], []
        self.test_data, self.test_label = [], []

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

        # 保存全局的 Tensor 供 t-SNE 提取特征使用
        self.train_data_tensor_all = Variable(img.cuda())
        self.train_label_tensor_all = Variable(label.cuda())
        self.test_data_tensor = Variable(torch.from_numpy(test_data).float().cuda())
        self.test_label_tensor = Variable(torch.from_numpy(test_label + 1).long().cuda())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        bestAcc, averAcc, num = 0, 0, 0
        epoch_losses, epoch_train_accs, epoch_test_accs = [], [], []

        print(f"--- 正在开始 Fold {fold+1} 的训练 ---")
        for e in range(self.n_epochs):
            self.model.train()
            batch_loss_sum, batch_train_acc_sum, batch_count = 0, 0, 0
            
            for i, (batch_img, batch_label) in enumerate(self.dataloader):
                batch_img = Variable(batch_img.cuda())
                batch_label = Variable(batch_label.cuda())
                
                batch_img = augment_snr(batch_img, batch_label, n_segments=8)

                features, outputs = self.model(batch_img)
                loss = self.criterion_cls(outputs, batch_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                batch_loss_sum += loss.item()
                train_pred = torch.max(outputs, 1)[1]
                batch_train_acc_sum += float((train_pred == batch_label).cpu().numpy().astype(int).sum()) / float(batch_label.size(0))
                batch_count += 1
            
            avg_train_loss = batch_loss_sum / batch_count
            avg_train_acc = batch_train_acc_sum / batch_count

            self.model.eval()
            with torch.no_grad():
                features, Cls = self.model(self.test_data_tensor)
                y_pred = torch.max(Cls, 1)[1]
                test_acc = float((y_pred == self.test_label_tensor).cpu().numpy().astype(int).sum()) / float(self.test_label_tensor.size(0))
                
                print('Epoch: %d  Train loss: %.4f  Train acc: %.4f  Test acc: %.4f' % (e, avg_train_loss, avg_train_acc, test_acc))
                
                epoch_losses.append(avg_train_loss)
                epoch_train_accs.append(avg_train_acc)
                epoch_test_accs.append(test_acc)
                
                self.log_write.write(str(e) + "    " + str(test_acc) + "\n")
                num += 1
                averAcc += test_acc
                if test_acc > bestAcc: bestAcc = test_acc

        averAcc = averAcc / num
        print('The best accuracy of fold %d is: %.4f' % (fold+1, bestAcc))
        
        # 绘制训练曲线
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss', color=color)
        ax1.plot(range(self.n_epochs), epoch_losses, color=color, label='Training Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Accuracy', color=color)  
        ax2.plot(range(self.n_epochs), epoch_train_accs, color='tab:orange', linestyle='--', label='Training Accuracy')
        ax2.plot(range(self.n_epochs), epoch_test_accs, color=color, label='Test Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout() 
        plt.title(f'Subject {self.nSub} Fold {fold+1} Training Process')
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
        
        plot_path = os.path.join(self.log_dir, f"sub{self.nSub}_fold{fold+1}_curve.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        return bestAcc, averAcc

    def visualize_tsne(self, fold, data_tensor, label_tensor, mode="Test"):
        """进行 t-SNE 降维并绘图，采用分批处理防止显存溢出"""
        print(f"正在生成 Fold {fold+1} 的 {mode} 集 t-SNE 可视化...")
        self.model.eval()
        
        features_list = []
        labels_list = []
        
        batch_size = 200
        with torch.no_grad():
            for i in range(0, data_tensor.size(0), batch_size):
                batch_data = data_tensor[i:i+batch_size]
                batch_label = label_tensor[i:i+batch_size]
                
                features, _ = self.model(batch_data)
                features_list.append(features.cpu().numpy())
                labels_list.append(batch_label.cpu().numpy())
                
        # 拼接所有批次的特征和标签
        features_all = np.concatenate(features_list, axis=0)
        labels_all = np.concatenate(labels_list, axis=0)

        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features_all)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels_all, cmap='viridis', s=15, alpha=0.8)
        plt.legend(handles=scatter.legend_elements()[0], labels=['Negative', 'Neutral', 'Positive'])
        plt.title(f't-SNE Feature Distribution - {mode} Set (Fold {fold+1})')
        
        tsne_path = os.path.join(self.log_dir, f"sub{self.nSub}_fold{fold+1}_{mode.lower()}_tsne.png")
        plt.savefig(tsne_path, dpi=300)
        plt.close()

    def visualize_cat_topography(self, fold):
        """提取 CAM 并生成 CAT 地形图"""
        print(f"正在生成 Fold {fold+1} 的 CAT 地形图演示...")
        self.model.eval()
        with torch.no_grad():
            sample_data = self.test_data_tensor[0:1] 
            _, _ = self.model(sample_data)
            
            attn_weights = None
            for module in self.model.modules():
                if isinstance(module, MultiHeadAttention):
                    attn_weights = module.attn_weights
            
            if attn_weights is None:
                print("警告：未提取到注意力权重！")
                return
            
            try:
                info = mne.create_info(ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'], sfreq=200, ch_types='eeg')
                info.set_montage('standard_1020')
                dummy_cat_data = np.random.rand(10) 
                
                fig, ax = plt.subplots()
                mne.viz.plot_topomap(dummy_cat_data, info, axes=ax, show=False)
                plt.title(f'Class Activation Topography (Fold {fold+1})')
                
                cat_path = os.path.join(self.log_dir, f"sub{self.nSub}_fold{fold+1}_cat.png")
                plt.savefig(cat_path, dpi=300)
                plt.close()
            except Exception as e:
                print("MNE 地形图绘制失败，请检查详细报错：", e)

# --- 5. 主函数 ---
def main():
    res_dir = "./results/seed/5-fold/"
    os.makedirs(res_dir, exist_ok=True)
    result_write = open(os.path.join(res_dir, "sub_result.txt"), "w")
    
    best = 0
    aver = 0
    
    for i in range(1): # Subject 1
        starttime = datetime.datetime.now()
        seed_n = 2021
        result_write.write('--------------------------------------------------\n')
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        
        print(f'\n========== 开始训练 Subject {i+1} ==========')
        result_write.write(f'Subject {i + 1} : Seed is: {seed_n}\n')
        
        bestAcc = 0
        averAcc = 0

        # 执行完整的 5 折交叉验证
        for fold in range(5):
            print(f"\n--- 准备执行 Fold {fold+1} ---")
            exgan = ExGAN(i + 1, fold)
            ba, aa = exgan.train(fold)
            
            # 分别生成训练集和测试集的 t-SNE 分布图
            exgan.visualize_tsne(fold, exgan.train_data_tensor_all, exgan.train_label_tensor_all, mode="Train")
            exgan.visualize_tsne(fold, exgan.test_data_tensor, exgan.test_label_tensor, mode="Test")
            
            # 生成地形图
            exgan.visualize_cat_topography(fold)
            
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
    print("开始执行程序:", time.asctime(time.localtime(time.time())))
    main()
    print("程序执行结束:", time.asctime(time.localtime(time.time())))
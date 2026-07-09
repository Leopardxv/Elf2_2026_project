"""
EEG-Conformer 8-Channel Inference Model
适配 best_fold1.pth 权重，去 DataParallel 前缀，纯推理模式
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, n_channels=8):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (n_channels, 1), (1, 1)),
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
        energy = torch.matmul(queries, keys.transpose(-2, -1))
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.matmul(att, values)
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


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size=40, n_classes=3, fc_in=280):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class EEGConformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=3, n_channels=8, fc_in=280):
        super().__init__(
            PatchEmbedding(emb_size, n_channels),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes, fc_in)
        )


def load_model(weight_path, device='cpu'):
    """Load 8-channel EEG-Conformer with best_fold1.pth weights"""
    model = EEGConformer(n_channels=8, n_classes=3, fc_in=280)
    state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    return model


if __name__ == '__main__':
    device = torch.device('cpu')
    model = load_model('/home/elf/Projects/emotions/best_fold1.pth', device)
    print(f'Model loaded successfully.')
    dummy = torch.randn(1, 1, 8, 200)
    with torch.no_grad():
        features, out = model(dummy)
    print(f'Input:  {list(dummy.shape)}')
    print(f'Output: {list(out.shape)} (logits)')
    print(f'Predicted class: {out.argmax(dim=1).item()}')
    total = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total:,}')

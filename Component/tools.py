from torch import nn
import copy
import torch
import numpy as np

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class AddNorm(nn.Module):
    def __init__(self, d_embed, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_embed, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, layer):
        # ?layer必须是以x为输入的函数
        layer_out = layer(x)
        return self.norm(x + self.dropout(layer_out))

# 这是自定义的LayerNorm实现，如果你想使用PyTorch内置的LayerNorm，可以直接用nn.LayerNorm
# 等效的，可以在此进行自定义修改
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        # 每个w的缩放系数
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 每个b的偏置的偏移量
        self.eps = eps
        # 防止除0错误

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
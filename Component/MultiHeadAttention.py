from torch import nn
import torch
import torch.nn.functional as F
import math
from Component.tools import clones
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_embed, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_embed % h == 0
        self.d_qkv = d_embed // h
        self.h = h
        self.linears = clones(nn.Linear(d_embed, d_embed), 4)
        # ?前三个是q,k,v的线性变换，最后一个是输出的线性变换
        self.attn_prob = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self,query, key, value, mask=None):
        d_qkv = query.size(-1)
        simlarity = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_qkv)
        if mask is not None:
            simlarity = simlarity.masked_fill(mask == 0, -1e9)
        attn_prob = F.softmax(simlarity, dim = -1)
        if self.dropout is not None:
            attn_prob = self.dropout(attn_prob)
        return torch.matmul(attn_prob, value), attn_prob
        
    def forward(self, query, key, value, mask=None):
        # if mask is not None:
        #     mask = mask.unsqueeze(1)
        no_batch = query.size(0)
        # !对qkv分别应用三个不同线性层，[n_batch, seq_len, d_embed] => [n_batch, h, seq_len, d_qkv]
        query, key, value = \
            [
                linear(qkv).view(no_batch, -1, self.h, self.d_qkv).transpose(1, 2)
                for linear, qkv in zip(self.linears, (query, key, value))
            ]
        # zip只把能配对的配对，多余的第四个忽略
        # !attention的经典公式计算 
        x, self.attn_prob = self.attention(query, key, value, mask=mask)
        # !把多头结果合并回去 [n_batch, h, seq_len, d_qkv] => [n_batch, seq_len, d_embed]
        x = x.transpose(1, 2).contiguous().view(no_batch, -1, self.h * self.d_qkv)
        # !最后再通过第四个线性层
        return self.linears[-1](x)


class AlibiAttention(MultiHeadedAttention):
    """
    ALiBi注意力，直接继承原MHA
    m是ALiBi的斜率，alibi_bias是距离偏置矩阵
    attention方法重载添加距离偏置
    """
    def __init__(self, h, d_embed, dropout=0.1, max_seq_len=8192):
        super().__init__(h, d_embed, dropout)
        m = torch.arange(1, h + 1, dtype=torch.float32)
        m = (-1.0 * torch.pow(5000.0, (m - 1) / (h - 1) * 8.0 / 10.0)).exp()
        self.register_buffer('m', m[None, :, None, None])
        # register_buffer就是模型中注册一个登记在册的常量
        # 且后续可以直接用self.m直接调用
        seq = torch.arange(max_seq_len, dtype=torch.float32)
        dist = seq[:, None] - seq[None, :]
        self.register_buffer('alibi_bias', self.m * dist)

    def attention(self, query, key, value, mask=None):

        d_qkv = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_qkv)
        
        # ALiBi核心：添加距离偏置
        tgt_len, src_len = scores.shape[-2:]
        # bias = self.alibi_bias[..., :tgt_len, :src_len].to(scores.dtype)
        bias = self.alibi_bias[:, :, :tgt_len, :src_len].to(dtype=scores.dtype, device=scores.device)
        scores += bias
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_prob = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            attn_prob = self.dropout(attn_prob)
        return torch.matmul(attn_prob, value), attn_prob


class SAibi1Attention(MultiHeadedAttention):
    """
    SAiBi1: 基于 ALiBi 的改进版本
    使用 head-wise 的 sigmoid 距离权重，对缩放点积打分做按元素乘法：
        Attention(i, j) = (Q_i K_j^T / sqrt(d)) ⊙ bias_ij
        bias_ij = σ(|i - j|), σ(x) = 1 / (1 + e^{-m_h x})
    """
    def __init__(self, h, d_embed, dropout=0.1, max_seq_len=8192):
        super().__init__(h, d_embed, dropout)
        # 与 AlibiAttention 相同的斜率构造方式，保持不同头的远近敏感度
        m = torch.arange(1, h + 1, dtype=torch.float32)
        m = (-1.0 * torch.pow(5000.0, (m - 1) / (h - 1) * 8.0 / 10.0)).exp()
        self.register_buffer('m', m[None, :, None, None])  # [1, h, 1, 1]

        # 预计算距离矩阵 |i - j|
        seq = torch.arange(max_seq_len, dtype=torch.float32)
        dist = torch.abs(seq[:, None] - seq[None, :])  # [L, L]
        self.register_buffer('dist', dist[None, None, ...])  # [1, 1, L, L]

    def attention(self, query, key, value, mask=None):
        d_qkv = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_qkv)  # [B, h, T, S]

        # SAiBi1: multiplicative sigmoid bias over distances
        tgt_len, src_len = scores.shape[-2:]
        dist = self.dist[..., :tgt_len, :src_len].to(dtype=scores.dtype, device=scores.device)
        bias = torch.sigmoid(self.m * dist)  # [1, h, T, S]
        scores = scores * bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_prob = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            attn_prob = self.dropout(attn_prob)
        return torch.matmul(attn_prob, value), attn_prob


class SAlibi2Attention(MultiHeadedAttention):
    """
    SAlibi2: 第二个改版 ALiBi，对缩放点积打分做加性 sigmoid 距离偏置：
        Attention(i,j) = Q_i K_j^T / sqrt(d) + bias_ij
        bias_ij = -m_h * σ(|i-j|),  σ(x) = 1 / (1 + e^x)
    """
    def __init__(self, h, d_embed, dropout=0.1, max_seq_len=8192):
        super().__init__(h, d_embed, dropout)
        # 与 AlibiAttention 相同的 head 斜率构造，决定不同头的远近敏感度
        m = torch.arange(1, h + 1, dtype=torch.float32)
        m = (-1.0 * torch.pow(5000.0, (m - 1) / (h - 1) * 8.0 / 10.0)).exp()
        self.register_buffer('m', m[None, :, None, None])  # [1, h, 1, 1]

        # 预计算 |i-j| 距离矩阵
        seq = torch.arange(max_seq_len, dtype=torch.float32)
        dist = torch.abs(seq[:, None] - seq[None, :])  # [L, L]
        self.register_buffer('dist', dist[None, None, ...])  # [1, 1, L, L]

    def attention(self, query, key, value, mask=None):
        d_qkv = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_qkv)  # [B, h, T, S]

        # SAlibi2: additive negative sigmoid distance bias
        tgt_len, src_len = scores.shape[-2:]
        dist = self.dist[..., :tgt_len, :src_len].to(dtype=scores.dtype, device=scores.device)
        # σ(x) = 1 / (1 + e^x) = sigmoid(-x)
        sigma = torch.sigmoid(-dist)  # [1, 1, T, S]
        bias = -self.m * sigma  # [1, h, T, S]，按 head 缩放
        scores = scores + bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_prob = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            attn_prob = self.dropout(attn_prob)
        return torch.matmul(attn_prob, value), attn_prob


class SAlibi4Attention(MultiHeadedAttention):
    """
    SAlibi4: 使用 log-sigmoid 距离的加性偏置版本：
        Attention(i,j) = Q_i K_j^T / sqrt(d) + bias_ij
        bias_ij = -m_h * θ(a + |i-j|)
        θ(x) = log(σ(x)) = -log(1 + e^x)
        σ(x) = 1 / (1 + e^x)
    其中 a 为可调超参数（偏移），用于控制整体衰减强度。
    """
    def __init__(self, h, d_embed, dropout=0.1, max_seq_len=8192, a: float = 1000.0):
        super().__init__(h, d_embed, dropout)
        # 与 Alibi 相同的 head 斜率构造
        m = torch.arange(1, h + 1, dtype=torch.float32)
        m = (-1.0 * torch.pow(5000.0, (m - 1) / (h - 1) * 8.0 / 10.0)).exp()
        self.register_buffer('m', m[None, :, None, None])  # [1, h, 1, 1]
        self.register_buffer('a', torch.tensor(float(a)))  # 标量超参数 a

        # 预计算 |i-j| 距离矩阵
        seq = torch.arange(max_seq_len, dtype=torch.float32)
        dist = torch.abs(seq[:, None] - seq[None, :])      # [L, L]
        self.register_buffer('dist', dist[None, None, ...])  # [1, 1, L, L]

    def attention(self, query, key, value, mask=None):
        d_qkv = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_qkv)  # [B, h, T, S]

        tgt_len, src_len = scores.shape[-2:]
        dist = self.dist[..., :tgt_len, :src_len].to(dtype=scores.dtype, device=scores.device)  # [1,1,T,S]
        x = self.a + dist
        theta = -F.softplus(x)             # θ(x) = -log(1 + e^x)
        bias = -self.m * theta             # bias_ij = -m_h * θ(x)
        scores = scores + bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_prob = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            attn_prob = self.dropout(attn_prob)
        return torch.matmul(attn_prob, value), attn_prob


class RoPEAttention(MultiHeadedAttention):
    """
    RoPE 注意力：在 Q、K 上应用旋转位置编码后再做缩放点积注意力。
    不依赖位置嵌入层，与 AlibiAttention 一样仅通过注意力内部注入位置信息。
    """
    def __init__(self, h, d_embed, dropout=0.1, max_seq_len=8192):
        super().__init__(h, d_embed, dropout)
        from Component.RoPE import RoPE
        self.rope = RoPE(self.d_qkv, max_seq_len=max_seq_len)

    def forward(self, query, key, value, mask=None):
        no_batch = query.size(0)
        query, key, value = [
            linear(qkv).view(no_batch, -1, self.h, self.d_qkv).transpose(1, 2)
            for linear, qkv in zip(self.linears, (query, key, value))
        ]
        query, key = self.rope(query, key)
        x, self.attn_prob = self.attention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous().view(no_batch, -1, self.h * self.d_qkv)
        return self.linears[-1](x)
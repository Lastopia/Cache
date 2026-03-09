"""
RoPE (Rotary Position Embedding) 组件
对 Q、K 按位置施加旋转，使注意力分数蕴含相对位置信息；V 不旋转。
与 ALiBi 一样不修改词嵌入，仅影响注意力层，便于作为可插拔组件使用。
"""
import torch


def apply_rope(q, k, cos, sin):
    """
    对 Q、K 应用旋转位置编码（按最后一维的 (0,1),(2,3),... 配对旋转）。
    q, k: [B, H, L, d]，d 为每头维度
    cos, sin: [L, d//2] 或可广播到该形状
    """
    d = q.size(-1)
    assert d % 2 == 0
    # [B, H, L, d] -> [B, H, L, d/2, 2]
    q0, q1 = q[..., 0::2], q[..., 1::2]
    k0, k1 = k[..., 0::2], k[..., 1::2]
    # cos/sin: [L, d/2] -> 与 q0 等对齐
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, d/2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot0 = q0 * cos - q1 * sin
    q_rot1 = q0 * sin + q1 * cos
    k_rot0 = k0 * cos - k1 * sin
    k_rot1 = k0 * sin + k1 * cos
    # 交错还原为 [B, H, L, d]
    q_rot = torch.stack([q_rot0, q_rot1], dim=-1).flatten(-2)
    k_rot = torch.stack([k_rot0, k_rot1], dim=-1).flatten(-2)
    return q_rot, k_rot


class RoPE(torch.nn.Module):
    """
    预计算并缓存 cos/sin [max_seq_len, d_qkv/2]，按序列长度截取后传给 apply_rope。
    不持有可学习参数，仅作为 RoPEAttention 的内部依赖。
    """

    def __init__(self, d_qkv, max_seq_len=8192, base=10000.0):
        super().__init__()
        self.d_qkv = d_qkv
        inv_freq = 1.0 / (base ** (torch.arange(0, d_qkv, 2, dtype=torch.float32) / d_qkv))
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, q, k, seq_len=None):
        """
        q, k: [B, H, L, d_qkv]
        返回 (q_rot, k_rot)，长度取 seq_len 或 q.size(2)
        """
        L = seq_len if seq_len is not None else q.size(2)
        cos = self.cos_cached[:L].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:L].to(dtype=q.dtype, device=q.device)
        return apply_rope(q, k, cos, sin)

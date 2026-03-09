from torch import nn
from Component.tools import clones
import Component.tools as tools

# ================== Model Structure =====================
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tar_embed, FC):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.tar_embed = tar_embed
        self.decoder = decoder
        self.FC = FC

    def forward(self, src, tar, src_mask, tar_mask):
        encoded = self.encoder(self.src_embed(src), src_mask)
        decoded = self.decoder(self.tar_embed(tar), encoded, src_mask, tar_mask)
        # return self.FC(decoded)
        # 说是用来预测下一个token的，所以decoded[:, :-1, :]，去掉最后一个时间步的输出
        return self.FC(decoded[:, :-1, :])

# ================== Encoder =====================
class EncoderLayer(nn.Module):
    def __init__(self, d_embed, attn, ffn, dropout):
        super().__init__()
        self.d_embed = d_embed
        self.attn = attn
        self.ffn = ffn
        self.attn_norm = tools.AddNorm(d_embed, dropout)
        self.ffn_norm = tools.AddNorm(d_embed, dropout)
    
    def forward(self, x, mask):
        x = self.attn_norm(x, lambda x: self.attn(x, x, x, mask))
        x = self.ffn_norm(x, self.ffn)
        # ?attn输入不是x,需要用lambda包装一下
        # ?ffn输入是x,可以直接传入                                  
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_embed, attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.d_embed = d_embed
        self.attn = attn      
        self.src_attn = src_attn  
        self.feed_forward = feed_forward

        self.attn_norm = tools.AddNorm(d_embed, dropout)
        self.src_attn_norm = tools.AddNorm(d_embed, dropout)
        self.ff_norm = tools.AddNorm(d_embed, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.attn_norm(x, lambda x: self.attn(x, x, x, tgt_mask))           # 自注意力
        x = self.src_attn_norm(x, lambda x: self.src_attn(x, m, m, src_mask))   # 交叉注意力
        x = self.ff_norm(x, self.feed_forward)                                  # FFN
        return x
    
class Encoder(nn.Module):
    def __init__(self, layer, N):  # layers → layer（单例克隆）
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.d_embed)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.d_embed)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

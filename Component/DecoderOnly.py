from torch import nn
from Component.tools import clones
from Component.tools import AddNorm

class DecoderOnly(nn.Module):
    def __init__(self, decoder, embed, FC):
        super().__init__()
        self.embed = embed
        self.decoder = decoder
        self.FC = FC  

    # 通常这里的mask是causal mask & padding mask的结合
    def forward(self, x, mask):  
        x = self.embed(x)
        x = self.decoder(x, mask)  
        return self.FC(x)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.d_embed)

    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_embed, attn, feed_forward, dropout):
        super().__init__()
        self.d_embed = d_embed
        self.attn = attn      # 只剩自注意力！
        self.feed_forward = feed_forward
        self.attn_norm = AddNorm(d_embed, dropout)
        self.ff_norm = AddNorm(d_embed, dropout)

    def forward(self, x, tgt_mask):
        x = self.attn_norm(x, lambda x: self.attn(x, x, x, tgt_mask))
        x = self.ff_norm(x, self.feed_forward)                        
        return x

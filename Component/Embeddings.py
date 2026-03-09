from torch import nn
import torch
import math

class WordEmbedding(nn.Module):
    def __init__(self, d_embed, vocab_size):
        super().__init__()
        self.d_embed = d_embed
        self.emb_tab = nn.Embedding(vocab_size, d_embed)
    
    def forward(self, x):
        return self.emb_tab(x) * math.sqrt(self.d_embed) 

class PositionalEmbedding(nn.Module):
    def __init__(self, d_embed, dropout=0.1, seq_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_emb = torch.zeros(seq_len, d_embed)
        pos = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        pos_emb[:, 0::2] = torch.sin(pos * div_term)
        pos_emb[:, 1::2] = torch.cos(pos * div_term)
        pos_emb = pos_emb.unsqueeze(0)
        self.register_buffer('pos_emb', pos_emb)
        
    def forward(self, x):
        return self.dropout(x+self.pos_emb[:, :x.size(1)])
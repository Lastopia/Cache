'''
N (int): 编码器和解码器的层数，模型深度
d_ff (int): 前馈网络的隐藏层维度 (升维降维用的)
h (int): 多头注意力机制的头数
'''
import copy
import torch
import torch.nn as nn
import numpy as np
import Component.EnDeCoder as ED
import Component.DecoderOnly as DO
import Component.MultiHeadAttention as MHA
import Component.FeedForwardNetwork as FFN
import Component.FullyConnection as FC
import Component.Embeddings as EB

c = copy.deepcopy

class SelfTransformer:
    def __init__(self,N=6, d_embed=512, d_ff=2048, h=8, dropout=0.1):
        self.N = N
        self.d_embed = d_embed
        self.d_ff = d_ff
        self.h = h
        self.dropout = dropout
        self.ffn = FFN.FFN(d_embed, d_ff, dropout)
    
    def pad_mask(self, input):
        return (input != 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T] ✅

    def sub_mask(self, input):
        """Causal mask，重载支持 Tensor 或 int"""
        if isinstance(input, int):
            seq_len = input
        else:
            seq_len = input.size(1)
        
        attn_shape = (1, seq_len, seq_len)
        subsequent = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent) == 0
    
    def ende_model(self,src_vocab_size,tar_vocab_size):
        attn = MHA.MultiHeadedAttention(self.h, self.d_embed)
        pos_emb = EB.PositionalEmbedding(self.d_embed,self.dropout)
        model = ED.EncoderDecoder(
            ED.Encoder(ED.EncoderLayer(self.d_embed, c(attn), c(self.ffn), self.dropout), self.N),
            ED.Decoder(ED.DecoderLayer(self.d_embed, c(attn), c(attn), c(self.ffn), self.dropout), self.N),
            nn.Sequential(EB.WordEmbedding(self.d_embed, src_vocab_size), c(pos_emb)),
            nn.Sequential(EB.WordEmbedding(self.d_embed, tar_vocab_size), c(pos_emb)),
            # EB.Embeddings(self.d_embed, src_vocab_size, self.dropout),
            # EB.Embeddings(self.d_embed, tar_vocab_size, self.dropout),
            FC.FC(tar_vocab_size,self.d_embed)
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        return model

    def deonly_model(self,vocab_size):
        attn = MHA.MultiHeadedAttention(self.h, self.d_embed)
        pos_emb = EB.PositionalEmbedding(self.d_embed,self.dropout)
        model = DO.DecoderOnly(
            DO.Decoder(DO.DecoderLayer(self.d_embed, c(attn), c(self.ffn), self.dropout),self.N),
            nn.Sequential(EB.WordEmbedding(self.d_embed, vocab_size), c(pos_emb)),
            FC.FC(vocab_size,self.d_embed)
        )
    
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def alibi_model(self, vocab_size):
        """ALiBi DecoderOnly：从原组件组合"""
        from Component.MultiHeadAttention import AlibiAttention
        from Component.Embeddings import WordEmbedding

        attn = AlibiAttention(self.h, self.d_embed, self.dropout)
        ffn = FFN.FFN(self.d_embed, self.d_ff, self.dropout)

        model = DO.DecoderOnly(
            DO.Decoder(DO.DecoderLayer(self.d_embed, c(attn), c(ffn), self.dropout), self.N),
            WordEmbedding(self.d_embed, vocab_size),  # 无位置！
            FC.FC(vocab_size, self.d_embed)
        )

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def salibi_model_one(self, vocab_size):
        """SAiBi1 DecoderOnly：使用改进的 ALiBi（sigmoid 距离权重，乘性偏置）"""
        from Component.MultiHeadAttention import SAibi1Attention
        from Component.Embeddings import WordEmbedding

        attn = SAibi1Attention(self.h, self.d_embed, self.dropout)
        ffn = FFN.FFN(self.d_embed, self.d_ff, self.dropout)

        model = DO.DecoderOnly(
            DO.Decoder(DO.DecoderLayer(self.d_embed, c(attn), c(ffn), self.dropout), self.N),
            WordEmbedding(self.d_embed, vocab_size),  # 保持与 ALiBi 相同，不显式位置编码
            FC.FC(vocab_size, self.d_embed)
        )

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def salibi_model_2(self, vocab_size):
        """SAlibi2 DecoderOnly：使用第二个改版 ALiBi（负 sigmoid 距离，加性偏置）"""
        from Component.MultiHeadAttention import SAlibi2Attention
        from Component.Embeddings import WordEmbedding

        attn = SAlibi2Attention(self.h, self.d_embed, self.dropout)
        ffn = FFN.FFN(self.d_embed, self.d_ff, self.dropout)

        model = DO.DecoderOnly(
            DO.Decoder(DO.DecoderLayer(self.d_embed, c(attn), c(ffn), self.dropout), self.N),
            WordEmbedding(self.d_embed, vocab_size),
            FC.FC(vocab_size, self.d_embed)
        )

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def salibi_model_4(self, vocab_size, a: float = 1000.0):
        """SAlibi4 DecoderOnly：log-sigmoid 距离加性偏置版本"""
        from Component.MultiHeadAttention import SAlibi4Attention
        from Component.Embeddings import WordEmbedding

        attn = SAlibi4Attention(self.h, self.d_embed, self.dropout, a=a)
        ffn = FFN.FFN(self.d_embed, self.d_ff, self.dropout)

        model = DO.DecoderOnly(
            DO.Decoder(DO.DecoderLayer(self.d_embed, c(attn), c(ffn), self.dropout), self.N),
            WordEmbedding(self.d_embed, vocab_size),
            FC.FC(vocab_size, self.d_embed)
        )

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def rope_model(self, vocab_size, max_seq_len=8192):
        """RoPE DecoderOnly：仅词嵌入 + 注意力内 RoPE，无显式位置编码层"""
        from Component.MultiHeadAttention import RoPEAttention
        from Component.Embeddings import WordEmbedding

        attn = RoPEAttention(self.h, self.d_embed, self.dropout, max_seq_len=max_seq_len)
        ffn = FFN.FFN(self.d_embed, self.d_ff, self.dropout)

        model = DO.DecoderOnly(
            DO.Decoder(DO.DecoderLayer(self.d_embed, c(attn), c(ffn), self.dropout), self.N),
            WordEmbedding(self.d_embed, vocab_size),
            FC.FC(vocab_size, self.d_embed)
        )

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

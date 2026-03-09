from torch import nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, vocab_size,d_embed=512):
        super(FC, self).__init__()
        self.proj = nn.Linear(d_embed, vocab_size)
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
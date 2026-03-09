from torch import nn
import torch.nn.functional as F

class FFN(nn.Module):
    '''
    位置感知前馈网络(FFN-PositionWise Feed Forward) 
    先升维，非线性，再降维
    论文中公式: FFN(x) = max(0, xW1 + b1)W2 + b2
    这里用ReLU作为激活函数
    扩维：学习更丰富的中间表示，增加模型参数(占总参数的70%)
    位置独立建模： 每个位置的FFN参数相同, 和位置无关 (PositonWise)
    d_ff=2048是平衡了表达力和计算量的结果, BERT有时候会用4*768维度
    Attention 是全局感知, FFN是局部感知
    '''
    def __init__(self, d_embed=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_embed, d_ff)
        self.w_2 = nn.Linear(d_ff, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
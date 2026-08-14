"""同构基线：GCN/SAGE/GATv2（目标类型子图）。"""
from .common import *

class HomoGCN(nn.Module):
    def __init__(self, din, dh, nc):
        super().__init__()
        self.c1, self.c2 = GCNConv(din, dh), GCNConv(dh, nc)
    def forward(self, x, ei):
        h = F.relu(self.c1(x, ei)); h = F.dropout(h, 0.5, self.training)
        return self.c2(h, ei)




class HomoSAGE(nn.Module):
    def __init__(self, din, dh, nc):
        super().__init__()
        self.c1, self.c2 = SAGEConv(din, dh), SAGEConv(dh, nc)
    def forward(self, x, ei):
        h = F.relu(self.c1(x, ei)); h = F.dropout(h, 0.5, self.training)
        return self.c2(h, ei)




class HomoGATv2(nn.Module):
    def __init__(self, din, dh, nc, heads=8):
        super().__init__()
        self.c1 = GATv2Conv(din, dh // heads, heads=heads)
        self.c2 = GATv2Conv(dh, nc, heads=1)
    def forward(self, x, ei):
        h = F.elu(self.c1(x, ei)); h = F.dropout(h, 0.5, self.training)
        return self.c2(h, ei)

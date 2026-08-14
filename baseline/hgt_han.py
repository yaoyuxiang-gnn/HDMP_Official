"""异构基线：HGT / HAN。"""
from .common import *


class HGTNet(nn.Module):
    def __init__(self, g, din, dh, nc, tgt):
        super().__init__()
        md = g.metadata()
        self.lin = nn.ModuleDict({nt: nn.Linear(g[nt].x.size(1), din) for nt in g.node_types})
        self.c1 = HGTConv(din, dh, md, heads=4)
        self.c2 = HGTConv(dh, dh, md, heads=4)
        self.out = nn.Linear(dh, nc)
        self.tgt = tgt
    def forward(self, xdict, eidict):
        h = {k: F.relu(self.lin[k](v)) for k, v in xdict.items()}
        h = self.c1(h, eidict); h = {k: F.relu(v) for k, v in h.items()}
        h = self.c2(h, eidict)
        return self.out(h[self.tgt])


class HANNet(nn.Module):
    def __init__(self, g, din, dh, nc, tgt):
        super().__init__()
        md = g.metadata()
        self.lin = nn.ModuleDict({nt: nn.Linear(g[nt].x.size(1), din) for nt in g.node_types})
        self.han = HANConv(din, dh, md, heads=4)
        self.out = nn.Linear(dh, nc)
        self.tgt = tgt
    def forward(self, xdict, eidict):
        h = {k: self.lin[k](v) for k, v in xdict.items()}
        h = self.han(h, eidict)
        return self.out(h[self.tgt])

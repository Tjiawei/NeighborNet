import torch, vit_pytorch
import torch.nn as nn
from torch.autograd import Variable

from model.base import GCNBlock,  Normalization, NeighborAttent, \
    BasicReason, FeedForward, GATv2, HopGCN, SelfAttention, MNeighborAttent, MBasicReason


class SimilarNet(nn.Module):
    def __init__(self, shot_dim, att_drop, seg_sz, topk, mode='self'):
        super(SimilarNet, self).__init__()
        if mode == 'self':
            # self.att_nei = NeighborAttent(shot_dim, dropout=att_drop, seg_sz=seg_sz, topk=topk)
            self.att_nei = MNeighborAttent(shot_dim, dropout=att_drop, seg_sz=seg_sz, topk=topk)
        elif mode == 'gat':
            self.att_nei = GATv2(shot_dim, shot_dim)
        elif mode == 'hopgnn':
            self.att_nei = HopGCN(shot_dim)

        self.relu = nn.ReLU()
        self.norm = Normalization(shot_dim, normalization='ln')

        self.ffc = FeedForward(shot_dim, int(shot_dim*1.5), drop=att_drop)
        self.ffc_norm = Normalization(shot_dim, normalization='ln')

    def forward(self, x, adj, inxs):
        adj = adj.squeeze()
        _x = x
        x = self.att_nei(x, adj, inxs)
        x = self.relu(x)
        x = self.norm(_x + x)

        _x = x
        x = self.ffc(x)
        x = self.ffc_norm(_x + x)

        return x

    def _build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask > 0, float(0))
        return mask

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class ReasonNet(nn.Module):
    def __init__(self, in_dim, seg_sz=20, topk=4, tnei=2, att_drop=0.1, mode='self'):
        super().__init__()
        if mode == 'self':
            # self.reason = BasicReason(in_dim, T=seg_sz, topk=topk, tnei=tnei, drop=att_drop)
            self.reason = MBasicReason(in_dim, T=seg_sz, topk=topk, tnei=tnei, drop=att_drop)
        elif mode == 'gat':
            self.reason = GATv2(in_dim, in_dim)
        elif mode == 'hopgnn':
            self.reason = HopGCN(in_dim)
        self.relu = nn.ReLU()
        self.att_norm = Normalization(in_dim, normalization='ln')

        self.ffc = FeedForward(in_dim, int(in_dim*1.5), drop=att_drop)
        self.ffc_norm = Normalization(in_dim, normalization='ln')

    def forward(self, x, radj, inxs):
        """
        :param x: (B, T, C)
        :return:
        """
        _x = x
        # mode = 'self
        x = self.reason(x, radj, inxs)

        x = self.relu(x)
        x = self.att_norm(_x+x)

        _x = x
        x = self.ffc(x)
        x = self.ffc_norm(_x+x)

        return x


class Baseline(nn.Module):
    def __init__(self, in_ch, bias=False, drop=0.2):
        super().__init__()
        self.gcn = GCNBlock(in_ch, in_ch, bias=bias)
        self.norm = Normalization(in_ch, normalization='ln')
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

        self.ffc = FeedForward(in_ch, in_ch * 2, drop=drop)
        self.ffc_norm = Normalization(in_ch, normalization='ln')

    def forward(self, x, adj):
        _x = x
        x = self.gcn(x, adj)
        x = self.drop(x)
        x = self.norm(_x + self.relu(x))

        _x = x
        x = self.ffc(x)
        x = self.ffc_norm(_x + x)

        return x


class Transformer(nn.Module):
    def __init__(self, in_ch, drop=0.5):
        super().__init__()
        self.attent = vit_pytorch.vit.Transformer(
            dim=in_ch,
            depth=2,
            heads=8,
            dim_head=in_ch//8,
            mlp_dim=int(in_ch*1.5),
            dropout=drop,
        )

    def forward(self, x):
        x = self.attent(x)

        return x


class FullyConnectGCN(nn.Module):
    def __init__(self, in_dim, dropout):
        super().__init__()
        self.gcn1 = SelfAttention(in_dim, dropout)
        self.norm_1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.gcn2 = SelfAttention(in_dim, dropout)
        self.norm_2 = nn.LayerNorm(in_dim, eps=1e-6)

    def forward(self, x):
        _x = x
        x = torch.relu(self.gcn1(x, x, x))
        x = self.norm_1(_x + x)

        _x = x
        x = torch.relu(self.gcn2(x, x, x))
        x = self.norm_2(_x + x)

        return x



import torch, einops
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange


class SelfAttention(nn.Module):
    def __init__(self, in_ch, dropout=0.1, is_scale=True):
        super(SelfAttention, self).__init__()

        out_ch = in_ch
        self.q = nn.Linear(in_ch, out_ch, bias=False)
        self.k = nn.Linear(in_ch, out_ch, bias=False)
        self.v = nn.Linear(in_ch, out_ch, bias=False)

        self.sqrt_dim = np.sqrt(out_ch)
        self.is_scale = is_scale

        self.final_drop = nn.Dropout(dropout)

        self.initialize_weight(self.q)
        self.initialize_weight(self.k)
        self.initialize_weight(self.v)

    def forward(self, ft_q, ft_k, ft_v, mask=None):
        """
        :param ft_q: (B, T, dim)
        :param ft_k: same
        :param ft_v: same
        :param mask: (B, T, T)
        :param reweight: (B, T, T)
        :return:
        """
        batch, T, _ = ft_k.shape
        q = self.q(ft_q)
        k = self.k(ft_k)
        v = self.v(ft_v)

        if self.is_scale:
            sim = torch.matmul(q, k.transpose(2, 1)) * self.sqrt_dim
        else:
            sim = torch.matmul(q, k.transpose(2, 1))
        if mask is not None:
            sim = sim + mask
        sim = F.softmax(sim, dim=-1)
        v = torch.matmul(sim, v)

        v = self.final_drop(v)

        return v

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class GAT(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1, is_scale=True):
        super().__init__()
        self.q = nn.Linear(in_ch, out_ch, bias=False)
        self.k = nn.Linear(in_ch, out_ch, bias=False)
        self.v = nn.Linear(in_ch, out_ch, bias=False)

        self.sqrt_dim = np.sqrt(out_ch)
        self.is_scale = is_scale ** -0.5

        self.final_drop = nn.Dropout(dropout)

        self.initialize_weight(self.q)
        self.initialize_weight(self.k)
        self.initialize_weight(self.v)

    def forward(self, ft_q, ft_k, ft_v, mask=None):
        """
        :param ft_q: (B, T, dim)
        :param ft_k: same
        :param ft_v: same
        :param mask: (B, T, T)
        :param reweight: (B, T, T)
        :return:
        """
        batch, T, _ = ft_k.shape
        q = self.q(ft_q)
        k = self.k(ft_k)
        v = self.v(ft_v)

        if self.is_scale:
            sim = torch.matmul(q, k.transpose(2, 1)) * self.sqrt_dim
        else:
            sim = torch.matmul(q, k.transpose(2, 1))
        if mask is not None:
            sim = sim + mask
        sim = F.softmax(sim, dim=-1)
        v = torch.matmul(sim, v)

        v = self.final_drop(v)

        return v

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class NeighborAttent(nn.Module):
    def __init__(self, in_ch, seg_sz=20, topk=4, dropout=0.1):
        super().__init__()
        self.seg_sz = seg_sz
        self.topk = topk

        out_ch = in_ch
        self.scale = out_ch ** (-0.5)

        self.q = nn.Linear(in_ch, out_ch, bias=False)
        self.k = nn.Linear(in_ch, out_ch, bias=False)
        self.qs = nn.Linear(in_ch, out_ch, bias=False)

        self.v = nn.Linear(in_ch, out_ch, bias=False)

        self.maxpool = nn.AdaptiveMaxPool2d((self.topk, 1))

        self.nn = nn.Linear(self.topk, 1, bias=False)
        self.ns = nn.Linear(self.topk, 1, bias=False)

        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.initialize_weight(self.q)
        self.initialize_weight(self.k)
        self.initialize_weight(self.v)
        self.initialize_weight(self.qs)
        self.initialize_weight(self.nn)
        self.initialize_weight(self.ns)

        nn.utils.weight_norm(self.nn, name='weight')
        nn.utils.weight_norm(self.ns, name='weight')

    def forward(self, x:torch.Tensor, adj, inxs):
        """
        :param x: (B, T, dim)
        :param adj: (B, T, T)
        :param inxs: (B, T, nei, T)
        :return:
        """
        B, T = x.shape[0], x.shape[1]
        mask = self.build_mask(adj)

        # # Version-2
        q = self.q(x)
        qs = self.qs(x)
        k = self.k(x)
        v = self.v(x)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        qs = F.normalize(qs, dim=-1)

        # single2single
        sim_s = torch.einsum('bxc, byc->bxy', (qs, qs))

        # neigh2neigh --- ij = sim(Nei(i), Nei(j))
        # qn, kn: (B, T ,N ,C)
        qn = torch.einsum('bxnt,bxtc -> bxnc', (inxs, q.unsqueeze(dim=1)))
        sim_nn = torch.einsum('biuxd, bujyd -> bijxy', (qn.unsqueeze(dim=2), qn.unsqueeze(dim=1)))
        sim_nn = sim_nn.reshape(B, -1, sim_nn.shape[-2], sim_nn.shape[-1])
        sim_nn = self.maxpool(sim_nn).squeeze()
        sim_nn = self.nn(sim_nn).squeeze()
        sim_nn = sim_nn.reshape(B, T, T)
        #
        # neigh2single --- ij = sim(Nei(i), j)
        kn = torch.einsum('bxnt,bxtc -> bxnc', (inxs, k.unsqueeze(dim=1)))
        # sim_ns: (B, T, T, N)
        sim_ns = torch.einsum('bxnc, byic->bxyni', (kn, k.unsqueeze(dim=2))).squeeze()
        sim_ns = self.ns(sim_ns).squeeze()

        sim = self.softmax(sim_s + sim_nn + sim_ns + mask)
        v = torch.einsum('bkt, btd->bkd', (sim, v))
        v = self.drop(v)
        return v

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask == 0, -float(1e22)).masked_fill_(mask>0, float(0))
        return mask


class Normalization(nn.Module):

    def __init__(self, embed_dim, timesz=20, k=3, normalization='batch'):
        super(Normalization, self).__init__()
        if normalization == 'batch':
            self.normalizer = nn.BatchNorm1d(embed_dim, affine=True)
        else:
            self.normalizer = nn.LayerNorm(embed_dim, eps=1e-6)

        self.type = normalization
        self.T = timesz

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.type == 'batch':
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        else:
            return self.normalizer(input)


class BasicReason(nn.Module):
    def __init__(self, in_ch, T=20, topk=4, tnei=2, drop=0.2):
        super().__init__()
        out_ch = in_ch
        self.pf = nn.Linear(in_ch, out_ch, bias=False)
        self.ns = nn.Linear(in_ch, out_ch, bias=False)
        # self.nn = nn.Linear(in_ch, out_ch, bias=False)
        self.scale = out_ch ** (-0.5)

        self.v = nn.Linear(in_ch, out_ch, bias=False)

        self.max_pf = nn.AdaptiveMaxPool2d((2*tnei+1, 1))
        self.agg_pf = nn.Linear(2*tnei+1, 1, bias=False)

        self.agg_ns = nn.Linear(topk, 1, bias=False)

        self.drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=-1)

        self.topk = topk
        self.tnei = tnei
        self.T = T

        self.ctxinx = self._tneigh_mask(mode='all')[None]

        self.initialize_weight(self.pf)
        self.initialize_weight(self.ns)
        # self.initialize_weight(self.nn)
        self.initialize_weight(self.v)
        self.initialize_weight(self.agg_pf)

        nn.utils.weight_norm(self.agg_pf, name='weight')
        nn.utils.weight_norm(self.agg_ns, name='weight')

    def forward(self, x:torch.Tensor, radj:torch.Tensor, inxs:torch.Tensor):
        """
        :param x: (B, T, C)
        :return:
            v: (B, T, C)
        """
        mask = self.build_mask(radj)
        B, T, _ = x.shape
        pf = self.pf(x)
        ns = self.ns(x)

        v = self.v(x)
        pf = F.normalize(pf, dim=-1)
        ns = F.normalize(ns, dim=-1)

        # # pre2j&future
        # # pre:B,T,2*N+1,C; fut:B,T,2*N+1,C, sim_pf:(B,T,T,2*N+1,2*N+1)
        ctx_q = torch.einsum('btnd, btdc->btnc', (self.ctxinx, pf.unsqueeze(dim=1)))
        ctx_k = torch.einsum('btnd, btdc->btnc', (self.ctxinx, pf.unsqueeze(dim=1)))
        sim_pf = torch.einsum('bxuic, buyjc->bxyij', (ctx_q.unsqueeze(dim=2), ctx_k.unsqueeze(dim=1)))

        sim_pf = sim_pf.reshape(B,-1,2*self.tnei+1, 2*self.tnei+1)
        sim_pf = self.max_pf(sim_pf).reshape(B, T, T, -1).squeeze()
        sim_pf = self.agg_pf(sim_pf).squeeze()

        # nei2single
        ns_q = torch.einsum('bxnt,bxtc -> bxnc', (inxs, ns.unsqueeze(dim=1)))
        sim_ns = torch.einsum('bxnc, byic->bxyni', (ns_q, ns.unsqueeze(dim=2))).squeeze()
        sim_ns = self.agg_ns(sim_ns).squeeze()

        sim = self.softmax(sim_pf+ sim_ns  + mask)
        v = torch.einsum('bkt, btd->bkd', (sim, v))
        v = self.drop(v)
        return v

    def _tneigh_mask(self,  mode='pre'):
        # v2: mask:(seg_sz, neigh+0/1, seg_sz)
        T = self.T
        tnei = self.tnei
        if mode == 'pre':
            mask = torch.zeros(T, tnei+1, T, requires_grad=False, device="cuda:0")
            neigh_inx = {}
            for i in range(T):
                if i - tnei < 0:
                    inx = [j for j in range(tnei+2) if j!=i]
                else:
                    inx = [j for j in range(i - tnei, i+1)]
                neigh_inx.update({i: inx})
        elif mode == 'fut':
            mask = torch.zeros(T, tnei+1, T, requires_grad=False, device="cuda:0")
            neigh_inx = {}
            for i in range(T):
                if i + 1 + tnei <= T:
                    inx = [j for j in range(i, i + tnei+1)]
                else:
                    inx = [j for j in range(T - tnei-1, T)]
                neigh_inx.update({i: inx})
        else:
            mask = torch.zeros(T, 2*tnei+1, T, requires_grad=False, device="cuda:0")
            neigh_inx = {}
            for i in range(T):
                if i - tnei < 0:
                    inx = [j for j in range(2*tnei+1)]
                elif i + 1 + tnei > T:
                    inx = [j for j in range(T - 2*tnei-1, T)]
                else:
                    inx = [j for j in range(i - tnei, i + tnei+1)]
                neigh_inx.update({i: inx})

        for i in neigh_inx.keys():
            for nn, j in enumerate(neigh_inx[i]):
                mask[i, nn, j] = 1
        return mask

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask == 0, -float(1e22)).masked_fill_(mask>0, float(0))
        return mask

    def initialize_weight(self, x):
        if isinstance(x, nn.Sequential):
            for module in x:
                if hasattr(module, 'weight'):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        else:
            nn.init.xavier_uniform_(x.weight)
            if x.bias is not None:
                nn.init.constant_(x.bias, 0)


class GATv2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=False)
        self.drop = nn.Dropout(0.5)

        self.proj_1 = nn.Linear(in_ch, out_ch, bias=False)
        self.proj_2 = nn.Linear(in_ch, out_ch, bias=False)
        self.proj_att = nn.Linear(out_ch, 1, bias=False)

    def forward(self, feat, adj):
        mask = self.build_mask(adj)
        T = feat.shape[1]
        feat_q = self.proj_1(feat)
        feat_k = self.proj_2(feat)
        feat_q = feat_q[:, None].repeat(1, T, 1, 1)
        feat_k = feat_k[:, :, None].repeat(1, 1, T, 1)
        att = F.relu(feat_q + feat_k)
        att = self.drop(att)
        att = self.proj_att(att)
        att = torch.squeeze(att, dim=-1)

        att = F.softmax(att + mask, dim=-1)

        feat_att = torch.matmul(att, feat)
        return feat_att

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask


class HopGCN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        hid_ch = in_ch*2
        self.drop = nn.Dropout(0.5)
        self.attent = GAT(hid_ch, in_ch, dropout=0.2)
        self.norm = nn.LayerNorm(in_ch)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        """
        :param x: (B, T, C)
        :param adj: (B, T, T)
        :return:
        """
        _x = x
        x_hop = torch.matmul(adj, x)
        x_hop = torch.concat((x, x_hop), dim=-1)

        # embedding
        x = self.attent(x_hop, x_hop, x_hop, self.build_mask(adj))
        x = _x + self.relu(x)
        x = self.norm(x)

        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask


# -------------  Basic  ------------- #


class GCNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False, is_drop=False):
        super(GCNBlock, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.linear = nn.Linear(self.in_ch, self.out_ch, bias=bias)

        self.is_drop = is_drop
        self.drop = nn.Dropout(0.2)

        self.initialize_weight(self.linear)

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

    def forward(self, feat, adj):
        """
        :param feat: (Batch, T, dim)
        :param adj:  (Batch, T, T)
        :return:
        """
        x = self.linear(feat)
        x = torch.matmul(adj, x)

        if self.is_drop:
            x = self.drop(x)

        return x


class TCN(nn.Module):
    def __init__(self, in_dim, out_dim, kernel:int=3):
        super().__init__()
        self.tcn = nn.Conv1d(in_dim, out_dim, kernel_size=kernel, padding=int(kernel//2), bias=False)
        self.relu = nn.ReLU()

        self.initialize_weight(self.tcn)

    def forward(self, x:torch.Tensor):
        x = x.transpose(-2, -1).contiguous()
        x = self.relu(self.tcn(x))
        return x.transpose(-2, -1).contiguous()

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class SineActivation(nn.Module):
    def __init__(self, in_features=1, out_features=256, n_shot=20):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

        self.pos_ids = torch.arange(n_shot, dtype=torch.float, device='cuda:0')[:, None]

    def forward(self, B):
        """
        :return:
        """
        v1 = self.f(torch.matmul(self.pos_ids, self.w) + self.b)
        v2 = torch.matmul(self.pos_ids, self.w0) + self.b0
        v = torch.cat([v1, v2], -1)

        v = einops.repeat(v, 't n -> b t n', b=B)
        return v


class FeedForward(nn.Module):
    def __init__(self, in_ch, hid_ch, drop=0.5):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(in_ch, hid_ch)
        self.linear_2 = nn.Linear(hid_ch, in_ch)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

        self.initialize_weight(self.linear_1)
        self.initialize_weight(self.linear_2)

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.drop(x)
        x = self.linear_2(x)

        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class RelativePosition(nn.Module):
    def __init__(self, scale, seg_sz=20):
        super().__init__()
        self.scale = scale
        self.seg_sz = seg_sz
        self.w = nn.parameter.Parameter(0.5*torch.ones(seg_sz, seg_sz))
        self.b = nn.parameter.Parameter(torch.randn(seg_sz, 1))
        q_idx = torch.arange(self.seg_sz, dtype=torch.long, device="cuda:0")
        self.rel_pos = (q_idx[None] - q_idx[:, None])**2

        # nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.b)

    def forward(self):
        rel_pos = torch.exp(-self.rel_pos * self.w + self.b)

        return rel_pos


class TCAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.q = nn.Conv1d(in_dim, in_dim, kernel_size=1, bias=False)
        self.k = nn.Conv1d(in_dim, in_dim, kernel_size=1, bias=False)
        self.gate = nn.Sigmoid()

        self.initialize_weight(self.q)
        self.initialize_weight(self.k)

    def forward(self, q:torch.Tensor, k:torch.Tensor):
        _q = q
        _k = k
        q = self.q(q.transpose(-2, -1).contiguous())
        k = self.k(k.transpose(-2, -1).contiguous())
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        gate = self.gate(q*k).transpose(-2, -1)
        out = _q + (1-gate)*_k

        return out

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class Embedding(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(in_ch, out_ch)
        self.linear_2 = nn.Linear(out_ch, out_ch)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

        self.initialize_weight(self.linear_1)
        self.initialize_weight(self.linear_2)

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.drop(x)
        x = self.linear_2(x)

        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

class MNeighborAttent(nn.Module):
    def __init__(self, in_ch, seg_sz=20, topk=4, dropout=0.1):
        super().__init__()
        self.seg_sz = seg_sz
        self.topk = topk

        out_ch = in_ch
        self.scale = out_ch ** (-0.5)

        self.q = nn.Linear(in_ch, out_ch, bias=False)
        self.k = nn.Linear(in_ch, out_ch, bias=False)
        self.qs = nn.Linear(in_ch, out_ch, bias=False)

        self.v = nn.Linear(in_ch, out_ch, bias=False)

        self.maxpool = nn.AdaptiveMaxPool2d((self.topk, 1))

        self.nn = nn.Linear(self.topk, 1, bias=False)
        self.ns = nn.Linear(self.topk, 1, bias=False)

        self.mha = MHA(in_ch, heads=1, dim_head = in_ch, dropout = dropout)

        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.initialize_weight(self.q)
        self.initialize_weight(self.k)
        self.initialize_weight(self.v)
        self.initialize_weight(self.qs)
        self.initialize_weight(self.nn)
        self.initialize_weight(self.ns)

        nn.utils.weight_norm(self.nn, name='weight')
        nn.utils.weight_norm(self.ns, name='weight')

    def forward(self, x:torch.Tensor, adj, inxs):
        """
        :param x: (B, T, dim)
        :param adj: (B, T, T)
        :param inxs: (B, T, nei, T)
        :return:
        """
        B, T = x.shape[0], x.shape[1]
        mask = self.build_mask(adj)

        # # Version-2
        q = self.q(x)
        qs = self.qs(x)
        k = self.k(x)
        v = self.v(x)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        qs = F.normalize(qs, dim=-1)

        # single2single
        sim_s = torch.einsum('bxc, byc->bxy', (qs, qs))

        # neigh2neigh --- ij = sim(Nei(i), Nei(j))
        # qn, kn: (B, T ,N ,C)
        qn = torch.einsum('bxnt,bxtc -> bxnc', (inxs, q.unsqueeze(dim=1)))
        sim_nn = torch.einsum('biuxd, bujyd -> bijxy', (qn.unsqueeze(dim=2), qn.unsqueeze(dim=1)))
        sim_nn = sim_nn.reshape(B, -1, sim_nn.shape[-2], sim_nn.shape[-1])
        sim_nn = self.maxpool(sim_nn).squeeze()
        sim_nn = self.nn(sim_nn).squeeze()
        sim_nn = sim_nn.reshape(B, T, T)
        #
        # neigh2single --- ij = sim(Nei(i), j)
        kn = torch.einsum('bxnt,bxtc -> bxnc', (inxs, k.unsqueeze(dim=1)))
        # sim_ns: (B, T, T, N)
        sim_ns = torch.einsum('bxnc, byic->bxyni', (kn, k.unsqueeze(dim=2))).squeeze()
        sim_ns = self.ns(sim_ns).squeeze()
        sim = self.softmax(sim_s + sim_nn + sim_ns + mask)

        v = torch.einsum('bkt, btd->bkd', (sim, v))
        v_mha = self.mha(x)
        v = self.drop(v+v_mha)

        # return v, self.softmax(sim_s+mask), sim
        return v

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask == 0, -float(1e22)).masked_fill_(mask>0, float(0))
        return mask


class MBasicReason(nn.Module):
    def __init__(self, in_ch, T=20, topk=4, tnei=2, drop=0.2):
        super().__init__()
        out_ch = in_ch
        self.pf = nn.Linear(in_ch, out_ch, bias=False)
        self.ns = nn.Linear(in_ch, out_ch, bias=False)
        # self.nn = nn.Linear(in_ch, out_ch, bias=False)
        self.scale = out_ch ** (-0.5)

        self.v = nn.Linear(in_ch, out_ch, bias=False)

        self.max_pf = nn.AdaptiveMaxPool2d((2*tnei+1, 1))
        self.agg_pf = nn.Linear(2*tnei+1, 1, bias=False)

        self.agg_ns = nn.Linear(topk, 1, bias=False)

        self.mha = MHA(in_ch, heads=1, dim_head=in_ch, dropout=drop)
        self.drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=-1)

        self.topk = topk
        self.tnei = tnei
        self.T = T

        self.ctxinx = self._tneigh_mask(mode='all')[None]

        self.initialize_weight(self.pf)
        self.initialize_weight(self.ns)
        # self.initialize_weight(self.nn)
        self.initialize_weight(self.v)
        self.initialize_weight(self.agg_pf)
        # self.initialize_weight(self.agg_ns)

        nn.utils.weight_norm(self.agg_pf, name='weight')
        nn.utils.weight_norm(self.agg_ns, name='weight')
        # nn.utils.weight_norm(self.agg_nn, name='weight')

    def forward(self, x:torch.Tensor, radj:torch.Tensor, inxs:torch.Tensor):
        """
        :param x: (B, T, C)
        :return:
            v: (B, T, C)
        """
        mask = self.build_mask(radj)
        B, T, _ = x.shape
        pf = self.pf(x)
        ns = self.ns(x)
        # nn = self.nn(x)

        v = self.v(x)
        pf = F.normalize(pf, dim=-1)
        ns = F.normalize(ns, dim=-1)
        # nn = F.normalize(nn, dim=-1)

        # # pre2j&future
        # # pre:B,T,2*N+1,C; fut:B,T,2*N+1,C, sim_pf:(B,T,T,2*N+1,2*N+1)
        ctx_q = torch.einsum('btnd, btdc->btnc', (self.ctxinx, pf.unsqueeze(dim=1)))
        ctx_k = torch.einsum('btnd, btdc->btnc', (self.ctxinx, pf.unsqueeze(dim=1)))
        sim_pf = torch.einsum('bxuic, buyjc->bxyij', (ctx_q.unsqueeze(dim=2), ctx_k.unsqueeze(dim=1)))

        sim_pf = sim_pf.reshape(B,-1,2*self.tnei+1, 2*self.tnei+1)
        sim_pf = self.max_pf(sim_pf).reshape(B, T, T, -1).squeeze()
        sim_pf = self.agg_pf(sim_pf).squeeze()

        # nei2single
        ns_q = torch.einsum('bxnt,bxtc -> bxnc', (inxs, ns.unsqueeze(dim=1)))
        sim_ns = torch.einsum('bxnc, byic->bxyni', (ns_q, ns.unsqueeze(dim=2))).squeeze()
        sim_ns = self.agg_ns(sim_ns).squeeze()

        sim = self.softmax(sim_pf+ sim_ns  + mask)
        v = torch.einsum('bkt, btd->bkd', (sim, v))

        v_mha = self.mha(x)
        v = self.drop(v+v_mha)
        return v
        # return v, self.softmax(mask+x.matmul(x.transpose(-2, -1))), sim

    def _tneigh_mask(self,  mode='pre'):
        # v2: mask:(seg_sz, neigh+0/1, seg_sz)
        T = self.T
        tnei = self.tnei
        if mode == 'pre':
            mask = torch.zeros(T, tnei+1, T, requires_grad=False, device="cuda:0")
            neigh_inx = {}
            for i in range(T):
                if i - tnei < 0:
                    inx = [j for j in range(tnei+2) if j!=i]
                else:
                    inx = [j for j in range(i - tnei, i+1)]
                neigh_inx.update({i: inx})
        elif mode == 'fut':
            mask = torch.zeros(T, tnei+1, T, requires_grad=False, device="cuda:0")
            neigh_inx = {}
            for i in range(T):
                if i + 1 + tnei <= T:
                    inx = [j for j in range(i, i + tnei+1)]
                else:
                    inx = [j for j in range(T - tnei-1, T)]
                neigh_inx.update({i: inx})
        else:
            mask = torch.zeros(T, 2*tnei+1, T, requires_grad=False, device="cuda:0")
            neigh_inx = {}
            for i in range(T):
                if i - tnei < 0:
                    inx = [j for j in range(2*tnei+1)]
                elif i + 1 + tnei > T:
                    inx = [j for j in range(T - 2*tnei-1, T)]
                else:
                    inx = [j for j in range(i - tnei, i + tnei+1)]
                neigh_inx.update({i: inx})

        for i in neigh_inx.keys():
            for nn, j in enumerate(neigh_inx[i]):
                mask[i, nn, j] = 1
        return mask

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask == 0, -float(1e22)).masked_fill_(mask>0, float(0))
        return mask

    def initialize_weight(self, x):
        if isinstance(x, nn.Sequential):
            for module in x:
                if hasattr(module, 'weight'):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        else:
            nn.init.xavier_uniform_(x.weight)
            if x.bias is not None:
                nn.init.constant_(x.bias, 0)

class MHA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)





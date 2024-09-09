import torch
import torch.nn as nn

from model.base import SineActivation, Embedding
from model.context import SimilarNet, ReasonNet
from model.detector import LatentDetector, BaSSLDet


class NNet(nn.Module):
    def __init__(self, shot_dim=1920, embed_dim=1920, att_drop=0.1, seg_sz=20, topk=4, tnei=2, mode='pretrain'):
        super().__init__()
        self.embed_pos = SineActivation(n_shot=seg_sz, out_features=256)
        self.proj = Embedding(shot_dim, embed_dim)
        shot_dim = embed_dim+256

        self.s1 = SimilarNet(shot_dim, att_drop, topk=topk, seg_sz=seg_sz, mode='self')
        self.r1 = ReasonNet(shot_dim, seg_sz=seg_sz, topk=topk, tnei=tnei, att_drop=att_drop, mode='self')

        self.relu = nn.ReLU()
        if mode == 'pretrain':
            self.detect = BaSSLDet(shot_dim, shot_dim//2, 1)
        else:
            # self.detect = Shotcol(shot_dim, shot_dim)
            self.detect = LatentDetector(seg_sz)

    def forward(self, x, adj, inxs):
        pos = self.embed_pos(x.shape[0])
        x = self.proj(x)
        x = torch.concat((x, pos), dim=-1)

        # cascade
        ctx = self.s1(x, adj[:, 0], inxs[:, 0])
        ctx = self.r1(ctx, adj[:, 1], inxs[:, 0])

        out = self.detect(ctx)
        return out.squeeze()




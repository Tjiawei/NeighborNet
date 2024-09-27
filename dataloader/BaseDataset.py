import numpy as np
import pickle as pkl
import random
import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, samplelist:list):
        self.samplist = samplelist
        random.shuffle(self.samplist)

    def __getitem__(self, ind):
        return self.samplist[ind]

    def __len__(self):
        return len(self.samplist)

    def _read_pkl(self, path):
        with open(path, 'rb') as f:
            data = pkl.load(f)
        return data

    def _build_graph(self, links:list, is_diag=True, is_norm=True):
        winsz = len(links)
        simgh = np.zeros((winsz, winsz))
        for i, link in enumerate(links):
            if len(link) != 0:
                simgh[i, link] = 1
            else:
                j = np.clip(i+1, 0, winsz-1)
                simgh[i, j] = 1
        if is_diag:
            simgh = np.eye(winsz) + simgh
        if is_norm:
            simgh = self._norm_graph(simgh)
        return simgh

    def _index_matric(self, links:list, n_top=5):
        T = len(links)
        inxmat = np.zeros((T, n_top, T))
        for i, link in enumerate(links):
            for j in range(n_top):
                inxmat[i, j, link[j]] = 1
            # inxmat[i, -1, i] = 1
        return inxmat

    def _gen_reason_link(self, links):
        r_link =[]
        for i, link in enumerate(links):
            if len(link) != 0:
                min_ed, max_ed = min(link), max(link)
                cand = list(range(min_ed, max_ed))
                cand = list(set(cand).difference(set(link)))
            else:
                cand = np.clip([i-1, i+1], 0, len(links)-1)
            r_link.append(cand)
        return r_link

    def _gen_rand_reason_link(self, links):
        r_link = []
        T = len(links)
        for i, link in enumerate(links):
            if len(link) != 0:
                cand = set(np.clip(range(i-5, i+5),0,T-1))
                cand = list(cand.difference(link))
            else:
                cand = np.clip([i - 1, i + 1], 0, len(links) - 1)
            r_link.append(cand)

        return r_link

    def _norm_graph(self, adj):
        """
        :param adj: (T, T)
        :return:
        """
        adj_sum = np.sum(adj, axis=-1)
        adj_sum = np.sqrt(1/adj_sum)
        adj_norm = adj_sum[:, None] * adj_sum[None, :] * adj
        return adj_norm



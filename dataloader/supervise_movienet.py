import numpy as np
import torch
import os
import json as js
import pickle as pkl
from dataloader.BaseDataset import BaseDataset


def read_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data


class MovienetDataset(BaseDataset):
    def __init__(self, samplelist:list, mode='train', topk=5):
        super(MovienetDataset, self).__init__(samplelist)
        self.mode = mode
        self.topk = topk

    def __getitem__(self, ind):
        data = self._read_pkl(self.samplist[ind])
        sample = data['data']
        label = data['label']

        if label != 1 and label !=0:
            label = 1
        sample = torch.from_numpy(sample)
        sample = sample.to(torch.float)
        alink = data['hop'][0]
        hop = []
        inxs = []
        gh = self._build_graph(alink)
        inx_gh = self._index_matric(alink, n_top=self.topk)
        rlink = self._gen_reason_link(alink)
        rgh = self._build_graph(rlink)
        hop.append(gh[None])
        hop.append(rgh[None])
        inxs.append(inx_gh[None])

        hop = np.concatenate(hop, axis=0)
        inxs = np.concatenate(inxs, axis=0)
        hop = torch.from_numpy(hop)
        inxs = torch.from_numpy(inxs).float()
        label = torch.from_numpy(np.array(label))
        label = label.to(torch.float)
        if self.mode == 'train':
            return sample, hop, inxs, label
        else:
            return self.samplist[ind], sample, hop, inxs, label


def gen_dataSet(ft_path, lb_path, gph_path, seg_sz=20, dim=2048, save_path=None):
    feats = read_pkl(ft_path)
    mnames = gen_labelName(lb_path)

    for name in tqdm(mnames):
        if name not in feats.keys():
            continue
        feat_m = feats[name]
        label_m = read_label(lb_path + '/' + name + '.txt')
        n_shot = len(label_m.keys())
        for c_id in range(n_shot):
            ctx, c_label  = sampleCtx(feat_m, label_m, c_id, seg_sz, dim=dim)

            shot_path = save_path + '/{}_shot{}.pkl'.format(name, c_id)
            hop_link= read_pkl(gph_path+ '/{}_shot{}.pkl'.format(name, c_id))['hop']
            sample = {'data': ctx, 'label': c_label, 'hop': hop_link}
            write_pkl(shot_path, sample)

        return 1

    return 1


def sampleCtx(data:dict, labels:dict, center_id, seg_sz, dim=2048):
    """
    Collecting shots centred in centre_id in the window at scale of seg_sz
    :param data:
    :param pairs:
    :param labels
    :param center_id:
    :param seg_sz: is a even number
    :return:
        ctx: (seg_sz, 2048)
        ctx_pair: (seg_sz, 5), second dim is shot index in a movie
        ctx_lab: (seg_sz,), same as above
        ctx_id: a list, where each elem. is shot's index in a movie
    """
    max_id = len(data.keys())
    half = seg_sz//2
    ctx_id = np.arange(center_id-half+1, center_id+half+1)
    ctx_id = np.clip(ctx_id, 0, max_id-1)

    if dim ==2048:
        ctx = np.zeros((seg_sz, dim))
    else:
        ctx = np.zeros((seg_sz, 5, dim//5))
    ctx_lab = []

    for i, shot_id in enumerate(ctx_id):
        ctx[i] = data[f'{shot_id:04d}'][None]
        if shot_id in labels.keys():
            ctx_lab.append(labels[shot_id])
        else:
            ctx_lab.append(0)

    if dim != 2048:
        ctx = np.reshape(ctx, (seg_sz, dim))
    ctx_lab = np.stack(ctx_lab)
    return ctx, ctx_lab[seg_sz//2-1]


def read_label(path):
    """
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        labelDict = {}
        while 1:
            line = f.readline()
            if not line:
                break
            line = line.split('\n')[0]
            shot_id, label = line.split(' ')
            labelDict.update({int(shot_id): int(label)})
    return labelDict

def gen_labelName(path):
    files = os.listdir(path)
    names = [file.split('.')[0] for file in files]

    return names

def read_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data


def write_pkl(path: str, data: dict):
    with open(path, 'wb') as f:
        pkl.dump(data, f)
    return 1

# class MovienetDataset_abl(BaseDataset):
#     def __init__(self, samplelist:list, mode='train', inx_abl=0):
#         super(MovienetDataset_abl, self).__init__(samplelist)
#         self.mode = mode
#         self.abl = inx_abl
#
#     def __getitem__(self, ind):
#         data = self._read_pkl(self.samplist[ind])
#         label = data['label']
#
#         if label != 1 and label !=0:
#             if self.mode == 'train':
#                 label = 1
#             else:
#                 label=0.5
#         links = data['hop'][self.abl]
#         half = len(links)//2
#         sample = data['data']
#         center = sample.shape[0]//2 - 1
#         sample = sample[center-half+1:center+half+1]
#         sample = torch.from_numpy(sample)
#         sample = sample.to(torch.float)
#         hop = []
#         inxs = []
#         gh = self._build_graph(links)
#         inx_gh = self._index_matric(links, n_top=len(links[0]))
#         rlink = self._gen_reason_link(links)
#         rgh = self._build_graph(rlink)
#         hop.append(gh[None])
#         hop.append(rgh[None])
#         inxs.append(inx_gh[None])
#
#         hop = np.concatenate(hop, axis=0)
#         inxs = np.concatenate(inxs, axis=0)
#         hop = torch.from_numpy(hop)
#         inxs = torch.from_numpy(inxs).float()
#         label = torch.from_numpy(np.array(label))
#         label = label.to(torch.float)
#         if self.mode == 'train':
#             return sample, hop, inxs, label
#         else:
#             return self.samplist[ind], sample, hop, inxs, label


def load_data(data_path, split_path, batch, mode='train', topk=5):
    with open(split_path, 'r') as f:
        data = js.load(f)
        trainSet = data['train'] + data['val']
        testSet = data['test']

    # testSet = read_pkl(split_path)
    # testlist = [data_path+'/'+name for name in testSet]

    samplelist = os.listdir(data_path)
    if mode == 'train':
        trainlist = [data_path+'/'+sample for sample in samplelist if sample.split('_')[0] in trainSet]
        trainDataset = MovienetDataset(trainlist, mode=mode, topk=topk)
        dataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch,
                                                  shuffle=True, drop_last=True, num_workers=4)
    if mode == 'test':
        testlist = [data_path+'/'+sample for sample in samplelist if sample.split('_')[0] in testSet]
        testDataset = MovienetDataset(testlist, mode=mode, topk=topk)
        dataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch,
                                                 shuffle=False, drop_last=False, num_workers=4)

    return dataLoader


def load_data_abl(data_path, split_path, batch, mode='train', inx_abl=0):
    with open(split_path, 'r') as f:
        data = js.load(f)
        trainSet = data['train'] + data['val']
        testSet = data['test']

    samplelist = os.listdir(data_path)
    if mode == 'train':
        trainlist = [data_path+'/'+sample for sample in samplelist if sample.split('_')[0] in trainSet]
        trainDataset = MovienetDataset_abl(trainlist, mode=mode, inx_abl=inx_abl)
        dataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch,
                                                  shuffle=True, drop_last=True, num_workers=1)
    if mode == 'test':
        testlist = [data_path+'/'+sample for sample in samplelist if sample.split('_')[0] in testSet]
        testDataset = MovienetDataset_abl(testlist, mode=mode, inx_abl=inx_abl)
        dataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch,
                                                 shuffle=False, drop_last=False, num_workers=1)

    return dataLoader


def load_transfer(data_path, batch):
    samplelist = os.listdir(data_path)
    datalist = [data_path+'/'+sample for sample in samplelist]
    Dataset = MovienetDataset(datalist, mode='test')
    dataLoader = torch.utils.data.DataLoader(Dataset, batch_size=batch,
                                             shuffle=False, drop_last=False, num_workers=4)
    return dataLoader


if __name__=='__main__':
    from tqdm import tqdm
    # data_path = r'C:\ResearchProject\Feature\sample_20'
    data_path = r'F:\OVSD\sample_20'
    files = os.listdir(data_path)
    samplist = [data_path+'/'+file for file in files]
    dataset = MovienetDataset(samplist)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=256,
                                             shuffle=True, drop_last=True, num_workers=4)
    for data in tqdm(dataLoader):
        label = data[2].data.numpy()
        is_r = (label>=0) * (label<=1)
        is_r = np.bool_(1 -is_r)
        if is_r.sum()>0:
            print(label[is_r])




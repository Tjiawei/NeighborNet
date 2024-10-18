import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from dataloader.supervise_movienet import load_data, load_transfer, load_data_abl
from model.NeighborNet import SLNet
from loss import bce, sigmoid_focal
from warm_up import warmup_decay_cosine
from metric import metric
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np

import pickle as pkl

torch.cuda.set_device(0)

import logging
import sys

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train_epoch(
        trainload,
        model,
        opti,
        lr_sh,
        gpu=0
):
    model.train()
    progress = tqdm(trainload)
    for i, sample in enumerate(progress):
        data, graphs, inxs, label = sample[0], sample[1], sample[2], sample[3]

        data = data.cuda(gpu)
        hop_gh = trans_graph(graphs, gpu)
        inxs = inxs.cuda(gpu)
        label = label.cuda(gpu)

        pred = model(data, hop_gh, inxs)
        loss = bce(pred, label)

        opti.zero_grad()
        loss.backward()
        opti.step()

        lr_sh.step()
        progress.set_postfix(loss=f'{loss.item():.8f}')

    return 1


def test_epoch(
        testload,
        model,
        need,
        gpu=0,
):
    predlist = []
    labelist = []
    pathlist = []
    scos = 0
    dcos = 0
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate((tqdm(testload))):
           paths, data, graphs, inxs, label = sample[0], sample[1], sample[2], sample[3], sample[4]

           data = data.cuda(gpu)
           hop_gh = trans_graph(graphs, gpu)
           inxs = inxs.cuda(gpu)

           pred = model(data, hop_gh, inxs)



           predlist.append(pred.data.cpu().numpy())
           labelist.append(label.data.cpu().numpy())
           pathlist.append(paths)
    met, moviePL = metric(pathlist, predlist, labelist, needs=need)
    return met, moviePL


# if i == 0:
#     scos = same_cos.data.cpu().numpy()
#     dcos = diff_cos.data.cpu().numpy()
# else:
#     scos = np.concatenate((scos, same_cos.data.cpu().numpy()), axis=-1)
#     dcos = np.concatenate((dcos, diff_cos.data.cpu().numpy()), axis=-1)

# def paint_cos_hist(scos, dcos):
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.hist(dcos, bins=30, facecolor='blue', edgecolor='black', alpha=0.6, label='different scene')
#     plt.hist(scos, bins=30, facecolor='green', edgecolor='black', alpha=0.6, label='same scene')
#     plt.xlabel('Cosine similarity', fontsize=12)
#     plt.legend(prop={'size':12})
#     plt.savefig(r'C:\ResearchProject\code\FBR\ShotLinker\wo_cos_dist.pdf')
#     plt.show()

    # return


def main(
        sample_path,
        split_path=None,
        batch=64,
        epoch=10,
        gpu=0,
        model_path=None,
        save_path=None,
):
    trainload = load_data(sample_path, split_path, batch, topk=5)
    testload = load_data(sample_path, split_path, 512, mode='test', topk=5)
    # trasferload = load_transfer(sample_path, 512)

    model = SLNet(2048, embed_dim=1024, att_drop=0.1, topk=5, seg_sz=20, tnei=2, mode='fine')
    # frozen = ['embed_pos', 's1']
    re_inin = 'detect'
    if model_path is not None:
        pretrain = torch.load(model_path, map_location='cpu')['state_dict']
        # new_weight = {}
        # for key in pretrain.keys():
        #     if re_inin not in key.split('.')[:2]:
        #         new_weight.update({key:pretrain[key]})
        # model.load_state_dict(new_weight, strict=False)
        model.load_state_dict(pretrain)
    model.cuda(gpu)

    if model_path is None:
        for para in model.parameters():
            para.requires_grad = True
    else:
        for name, para in model.named_parameters():
            if name.split('.')[0] in re_inin:
                para.requires_grad = True
            else:
                para.requires_grad = False

    max_miou = 0
    max_map = 0
    opti = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4)
    iter_num = len(trainload)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        opti,
        warmup_decay_cosine(iter_num, iter_num * (epoch - 1))
    )

    for i in range(epoch):
        train_epoch(trainload, model, opti, lr_scheduler, gpu)
        met, moviePL = test_epoch(testload, model, ['map', 'miou', 'f1'], gpu=gpu)

        # save model
        if save_path is not None:
            if max_miou < met['mIoU'] or max_map < met['mAP']:
                max_miou = met['mIoU']
                max_map = met['mAP']
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'miou': max_miou,
                    'map': max_map,
                    'f1':met['F1'],
                    'optim': opti.state_dict()
                },
                    save_path + '/epoch_{}.pth.tar'.format(i+1)
                )
        #
        # if i==1:
        #     for para in model.parameters():
        #         para.requires_grad = True
        #
        # # print('\n')
        print('{} epoch: mAP:{:.3f}, mIoU:{:.3f}'.format(i+1, met['mAP'], met['mIoU']))
        print('{} epoch: F1:{:.3f}'.format(i + 1, met['F1']))
        # print('{} epoch: mAP:{:.3f}'.format(i+1, met['mAP']))
        # print('{} epoch: F1:{:.3f}'.format(i + 1, met['F1']))

    return 1


def main_abl(
        sample_path,
        split_path=None,
        batch=64,
        epoch=10,
        gpu=0,
        abl_inx=0,
        abl_num=0,
        log_name=''
):
    # trainload = load_data_abl(sample_path, split_path, batch, inx_abl=abl_inx)
    # testload = load_data_abl(sample_path, split_path, 512, mode='test', inx_abl=abl_inx)
    trainload = load_data(sample_path, split_path, batch)
    testload = load_data(sample_path, split_path, 512, mode='test')

    model = SLNet(2048, embed_dim=abl_num, att_drop=0.2, win_size=5, topk=5, seg_sz=20, tnei=2, mode='train')
    model.cuda(gpu)

    for para in model.parameters():
        para.requires_grad = True

    opti = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4)
    iter_num = len(trainload)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        opti,
        warmup_decay_cosine(iter_num, iter_num * (epoch - 1))
    )
    record = []
    for i in range(epoch):
        train_epoch(trainload, model, opti, lr_scheduler, gpu)
        met, _ = test_epoch(testload, model, ['map', 'miou', 'f1'], gpu=gpu)

        record.append([i, met['mAP'], met['mIoU'], met['F1']])
        # if i >=int(0.75*epoch):
        #     break

        print('{} epoch: mAP:{:.3f}, mIoU:{:.3f}'.format(i+1, met['mAP'], met['mIoU']))
        print('{} epoch: F1:{:.3f}'.format(i + 1, met['F1']))

    stdout_backup = sys.stdout
    log_file = open(log_name, 'w')
    sys.stdout = log_file
    for i in range(len(record)):
        print('{} epoch: mAP:{:.3f}, mIoU:{:.3f}, F1:{:.3f}'.format(record[i][0], record[i][1], record[i][2], record[i][3]))
    log_file.close()
    sys.stdout = stdout_backup
    return 1


def adjust_lr(optimizer):
    for param in optimizer.param_groups:
        param['lr'] *= 0.1
    return 1


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def write_pkl(path: str, data: dict):
    with open(path, 'wb') as f:
        pkl.dump(data, f)
    return 1


# ------------- Utilize -------------  #
def trans_graph(graph, gpu):
    graph = graph.to(torch.float)
    graph = graph.cuda(gpu)
    graph = Variable(graph, requires_grad=False)
    return graph


if __name__=='__main__':
    data_path = r'G:\MovieNet\NeighborNet\sample_20'
    # data_path = r'G:\MovieNet\NeighborNet\vit_20\train_test'
    # data_path = r'F:\OVSD\sample_20'
    split_path = r'F:\MovieNet\Features\split318.json'
    # split_path = r'G:\MovieNet\NeighborNet\similar_scn.pkl'
    save_path = r'C:\ResearchProject\code\FBR\ShotLinker\model_zoo\open_supervise'
    model_path = r'C:\ResearchProject\code\FBR\ShotLinker\model_zoo\supervised_abl\epoch_3.pth.tar'

    # ablation studies

    _ = main(data_path, split_path, batch=512, epoch=10, save_path=save_path, model_path=None)
    # with open(r'C:\ResearchProject\code\FBR\ShotLinker\Results\ours_wofc_result.pkl', 'wb') as f:
    #     pkl.dump(moviePL, f)

    # abl = [1280]
    # for i in range(len(abl)):
    #     print('abla_num:{}'.format(abl[i]))
    #     main_abl(data_path, split_path, batch=512, epoch=10, abl_inx=i, abl_num=abl[i],
    #              log_name=r'C:\ResearchProject\code\FBR\ShotLinker\Results\log_dim{}.log'.format(abl[i]))

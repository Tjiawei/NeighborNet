import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weighter(label, weight=[0.5, 5]):
    batch_weight = weight[0] * (1 - label) + weight[1] * label
    return batch_weight


def bce(pred, label):
    batch_weight = weighter(label)
    loss = F.binary_cross_entropy(pred, label, batch_weight)
    # loss = F.binary_cross_entropy(pred, label)
    return loss


def sigmoid_focal(pred, label, alpha=0.9, gamma=2):
    ce_loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
    p_t = pred*label + (1-pred)*(1-label)
    loss = ce_loss * ((1-p_t)**gamma)

    if alpha>=0:
        alpha_t = alpha*label+(1-alpha)*(1-label)
        loss = alpha_t*loss

    return loss.mean()


def pseudo_bce(pred, label):
    weight = weighter(label, [0.5, 5])
    ce_loss = F.binary_cross_entropy_with_logits(pred, label, weight, reduction='none')

    reg = torch.mean(pred * pred)
    return ce_loss.mean() + 0.3*reg


def bce_v2(pred, label_ind):
    b = pred.shape[0]
    labels = torch.zeros(b * 2, dtype=torch.float, device=pred.get_device())
    labels[:b] = 1
    batch_ind = torch.arange(b, device=pred.get_device())
    pos_ind, neg_ind = torch.split(label_ind, [1, 1], dim=1)

    pred = torch.cat([pred[batch_ind, pos_ind[:, 0]], pred[batch_ind, neg_ind[:, 0]]], dim=0)

    loss = F.binary_cross_entropy(pred, labels, reduction='mean')

    reg = torch.mean(pred*pred)
    return loss + 0.8*reg



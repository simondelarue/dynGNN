# -*- coding: utf-8 -*-

from os import dup
import numpy as np

from abc import ABC, abstractmethod
from numpy.lib.twodim_base import tri

import torch
import torch.nn.functional as F

import torchsort

class Loss(ABC):
    ''' Abstract class for loss. '''   

    def __init__(self, loss_name: str):
        self.loss_name = loss_name

    @abstractmethod
    def compute(self, pos_score, neg_score, device, sg, timestep,):
        pass


class BCEWithLogitsLoss(Loss):

    def compute(self, pos_score, neg_score, device, **kwargs):
        ''' Computes BCE with logits loss. '''

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)

        return F.binary_cross_entropy_with_logits(scores, labels)


class GraphSageLoss(Loss):

    def compute(self, pos_score, neg_score, device, **kwargs):
        ''' Computes loss implemented in GraphSage paper. '''

        pos = F.binary_cross_entropy_with_logits(pos_score, torch.ones(pos_score.shape[0]).to(device), reduction='mean')
        neg = F.binary_cross_entropy_with_logits(neg_score, torch.zeros(neg_score.shape[0]).to(device), reduction='mean')

        return torch.sum(pos) + 1 * torch.sum(neg)    


class MarginRankingLoss(Loss):

    def compute(self, pos_score, neg_score, device, **kwargs):
        ''' Computes margin ranking loss. '''

        n_min = min(len(pos_score), len(neg_score))

        return (1 - pos_score[:n_min].unsqueeze(1) + neg_score[:n_min].view(n_min, -1)).clamp(min=0).mean()


class TorchMaringRankingLoss(Loss):

    def compute(self, pos_score, neg_score, device, **kwargs):
        ''' Computes margin ranking loss using torch module. '''

        x1 = torch.cat([pos_score, neg_score])
        x2 = torch.cat([neg_score, pos_score])
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)

        return F.margin_ranking_loss(x1, x2, labels, margin=1.0)


class PairwiseLoss(Loss):

    def compute(self, pos_score, neg_score, device, method='soft-rank', **kwargs):
        ''' Computes pairwise loss :
            
            :math:`\sum_{s=1}^{E-1} \sum_{\{i: l(i) < l(s)\}} \phi(f(x_s) - f(x_i))`
            
            where 
            * :math:`l(i)` denotes the label, i.e ranking of edge :math:`x_{i}`. Thus, if :math:`l(i)>l(s)`, :math:`x_i` should be ranked before :math:`x_s`.
            * :math:`f(x_i)` a ranking function that should be learned. 
            * :math:`\phi` a smoothing operator. `softmax` is used here. 
            * :math:`E` the number of edges. '''
        
        sg = kwargs['sg']
        timestep = kwargs['timestep']

        src, dest, ranks, dup_mask = sg.rank_edges(sg.data_df, sg.trange_train, metric=None, timestep=timestep)

        if len(pos_score) != len(ranks):
            pos_score = pos_score[:int(len(ranks))]
            pred_ranks = torch.argsort(-pos_score[~dup_mask])
        else:
            pred_ranks = torch.argsort(-pos_score[~dup_mask])

        true_ranks = torch.tensor(np.array(range(0, len(pred_ranks))))

        loss = 0 * torch.sum(-pos_score)

        if method == 'values':
            for i, elem in enumerate(true_ranks[:-1]):
                v = -pos_score[~dup_mask][i] * torch.ones(len(true_ranks) - i - 1)
                loss = loss + torch.sigmoid(torch.sum(v -(-pos_score[~dup_mask][(i+1):])))
        
        elif method == 'soft-rank':
            pred_soft_ranks = torchsort.soft_rank(-pos_score[~dup_mask].unsqueeze(0)).squeeze(0)
            true_soft_ranks = torchsort.soft_rank(true_ranks.unsqueeze(0)).squeeze(0)

            # Vectorized way : note that true_ranks need to be ordered
            '''l = len(pred_soft_ranks) - 1
            triu = torch.triu(torch.ones((l, l)))

            x1 = torch.diag(pred_soft_ranks[:-1]).matmul(triu)
            x2 = triu.matmul(torch.diag(pred_soft_ranks[1:]))

            loss = torch.sum(torch.sigmoid(x1 - x2))'''

            # Linear way
            for i, elem in enumerate(true_soft_ranks):
                v = pred_soft_ranks[i] * torch.ones(len(true_soft_ranks) - i - 1)
                loss = loss + torch.sigmoid(torch.sum(v -(pred_soft_ranks[(i+1):])))

        return loss


class SpearmanrLoss(Loss):

    def compute(self, pos_score, neg_score, device, sg, timestep, **kw):

        #sg = kwargs['sg']
        #timestep = kwargs['timestep']

        src, dest, ranks, dup_mask = sg.rank_edges(sg.data_df, sg.trange_train, metric=None, timestep=timestep)

        if len(pos_score) != len(ranks):
            pos_score = pos_score[:int(len(ranks))]
            #pred_ranks = torch.argsort(-pos_score[~dup_mask])
            pred_ranks = torchsort.soft_rank(pos_score[~dup_mask].unsqueeze(0), **kw)
        else:
            pred_ranks = torchsort.soft_rank(pos_score[~dup_mask].unsqueeze(0), **kw)

        true_ranks = torch.tensor(np.array(range(0, len(pred_ranks.squeeze(0)))))
        target = torchsort.soft_rank(true_ranks.unsqueeze(0), **kw)

        pred_ranks = pred_ranks - pred_ranks.mean()
        pred_ranks = pred_ranks / pred_ranks.norm()
        target = target - target.mean()
        target = target / target.norm()

        return (pred_ranks * target).sum()


def loss_factory(loss_name: str) -> Loss:
    ''' Instanciate appropriate loss class according to input. '''
    
    if loss_name == 'BCEWithLogits':
        loss = BCEWithLogitsLoss(loss_name)
    elif loss_name == 'marginRanking':
        loss = MarginRankingLoss(loss_name)
    elif loss_name == 'graphSage':
        loss = GraphSageLoss(loss_name)
    elif loss_name == 'torchMarginRanking':
        loss = TorchMaringRankingLoss(loss_name)
    elif loss_name == 'pairwise':
        loss = PairwiseLoss(loss_name)
    elif loss_name == 'spearmanr':
        loss = SpearmanrLoss(loss_name)
    
    return loss
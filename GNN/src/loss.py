# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

class Loss(ABC):
    ''' Abstract class for loss. '''   

    def __init__(self, loss_name: str):
        self.loss_name = loss_name

    @abstractmethod
    def compute(self, pos_score, neg_score, device):
        pass


class BCEWithLogitsLoss(Loss):

    def compute(self, pos_score, neg_score, device):
        ''' Computes BCE with logits loss. '''

        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)

        return F.binary_cross_entropy_with_logits(scores, labels)


class GraphSageLoss(Loss):

    def compute(self, pos_score, neg_score, device):
        ''' Computes loss implemented in GraphSage paper. '''

        pos = F.binary_cross_entropy_with_logits(pos_score, torch.ones(pos_score.shape[0]).to(device), reduction='mean')
        neg = F.binary_cross_entropy_with_logits(neg_score, torch.zeros(neg_score.shape[0]).to(device), reduction='mean')

        return torch.sum(pos) + 1 * torch.sum(neg)    


class MarginRankingLoss(Loss):

    def compute(self, pos_score, neg_score, device):
        ''' Computes margin ranking loss. '''

        n_min = min(len(pos_score), len(neg_score))

        return (1 - pos_score[:n_min].unsqueeze(1) + neg_score[:n_min].view(n_min, -1)).clamp(min=0).mean()

class TorchMaringRankingLoss(Loss):

    def compute(self, pos_score, neg_score, device):
        ''' Computes margin ranking loss using torch module. '''

        x1 = torch.cat([pos_score, neg_score])
        x2 = torch.cat([neg_score, pos_score])
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)

        return F.margin_ranking_loss(x1, x2, labels, margin=1.0)


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
    
    return loss
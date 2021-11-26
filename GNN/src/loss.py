# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

def compute_loss(pos_score, neg_score, device):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

# Simplify loss not to consider negative sampling
# This is done to try a simpler PDAG structure
def compute_loss_simp(pos_score, device):
    scores = pos_score
    labels = torch.ones(pos_score.shape[0]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)


def graphSage_loss(pos_score, neg_score, device):
    pos = F.binary_cross_entropy_with_logits(pos_score, torch.ones(pos_score.shape[0]).to(device), reduction='mean')
    neg = F.binary_cross_entropy_with_logits(neg_score, torch.zeros(neg_score.shape[0]).to(device), reduction='mean')
    return torch.sum(pos) + 1 * torch.sum(neg)    
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report

def compute_kendall(true_ranks, pred_ranks, weighted=True):
    ''' Kendall tau given two lists of ranks. If weighted is True, Kendall tau is weighted, i.e gives more
        importance to rank differences when ranks are higher. '''
        
    if weighted:
        tau, p_value = stats.weightedtau(true_ranks, pred_ranks)
    else:
        tau, p_value = stats.kendalltau(true_ranks, pred_ranks)

    return tau, p_value

def compute_auc(pos_score, neg_score):
    
    # Compute auc
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    
    # Compute fpr and tpr
    fpr, tpr, _ = roc_curve(labels, scores)
    
    return roc_auc_score(labels, scores), fpr, tpr

def compute_f1_score(pos_score, neg_score, average):

    # F1 Score
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()

    return f1_score(labels, scores, average)

def compute_classif_report(pos_score, neg_score):

    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()

    return classification_report(labels, scores, digits=3)
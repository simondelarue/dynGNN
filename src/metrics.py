# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from scipy import stats
import os

import torch
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, classification_report
from sklearn.utils import shuffle

from utils import write_log

def compute_metric(metric, **kwargs):
    ''' Compute performance metric between true values and predicted values, according to `metric`. '''

    history = {}

    if metric == 'auc':
        pos_score = kwargs['pos_score']
        neg_score = kwargs['neg_score']

        auc, fpr, tpr = compute_auc(pos_score, neg_score)
        # Save results
        history[f'test_{metric}'] = auc
        history['test_fpr'] = fpr
        history['test_tpr'] = tpr

    elif metric == 'f1_score':
        pos_score = kwargs['pos_score']
        neg_score = kwargs['neg_score']

        score = compute_f1_score(pos_score, neg_score, 'macro')
        history[f'test_{metric}'] = score
        
    elif ('kendall' in metric) or ('spearmanr' in metric):

        LOG_PATH = f'{os.getcwd()}/logs'

        sg = kwargs['sg']
        timestep = kwargs['timestep']
        pos_score = kwargs['pos_score']
        predictor = str(kwargs['predictor'])
        feat_struct = kwargs['feat_struct']
        model_name = kwargs['model_name']
        shuffle_test = kwargs['shuffle_test']

        # True ranks
        src, dest, ranks, dup_mask = sg.rank_edges(sg.data_df, sg.trange_val, metric=metric, timestep=timestep)

        # Predicted ranks
        if len(pos_score) != len(ranks):
            len_pos_score = len(pos_score)
            pos_score = pos_score[:int(len(ranks))]
            pred_ranks = np.argsort(-pos_score[~dup_mask])
        else:
            pred_ranks = np.argsort(-pos_score[~dup_mask])

        true_ranks = np.array(range(0, len(pred_ranks)))

        # Shuffle test links true order (in order to verify that results are not randomly good for
        # temporal structures).
        if shuffle_test == 'True':
            true_ranks = shuffle(true_ranks)


        # ------ Save results in df for analysis
        
        data = {
            'dataset': [sg.name] * len(true_ranks),
            'feat_struct': [feat_struct] * len(true_ranks),
            'model_name': [model_name] * len(true_ranks),
            'metric': [metric] * len(true_ranks),
            'predictor': [predictor] * len(true_ranks),
            'loss_func': ['MRL_PWL'] * len(true_ranks),
            'src': src[~dup_mask],
            'dest': dest[~dup_mask],
            'true_ranks': true_ranks,
            'pred_ranks': pred_ranks
        }
        df_analysis = pd.DataFrame(data)
        if shuffle_test == 'True':
            df_analysis.to_pickle(f'{LOG_PATH}/SF2H/preds_{feat_struct}_{model_name}_{metric}_shuffled.pkl', protocol=3)
        else:
            df_analysis.to_pickle(f'{LOG_PATH}/SF2H/preds_{feat_struct}_{model_name}_{metric}.pkl', protocol=3)
        # ------ 
        
        if metric.startswith('wkendall'):
            tau, _ = compute_kendall(true_ranks, pred_ranks, weighted=True)
            history['test_wkendall'] = tau

        elif metric.startswith('kendall'):
            tau, p_value = compute_kendall(true_ranks, pred_ranks, weighted=False)
            history[f'test_{metric}'] = tau
            # Save p-values in Log
            txt = f'{sg.name}, {predictor}, {feat_struct}, {metric}, {tau}, {p_value}\n'
            write_log(f'{LOG_PATH}/p_values_{metric}.txt', txt)

        elif metric.startswith('spearmanr'):
            rho, p_value = compute_spearmanr(true_ranks, pred_ranks)
            history[f'test_{metric}'] = rho
            # Save p-values in Log
            txt = f'{sg.name}, {predictor}, {feat_struct}, {metric}, {rho}, {p_value}\n'
            write_log(f'{LOG_PATH}/p_values_{metric}.txt', txt)

    return history

def compute_kendall(true_ranks, pred_ranks, weighted=True):
    ''' Kendall tau given two lists of ranks. If weighted is True, Kendall tau is weighted, i.e gives more
        importance to rank differences when ranks are higher. '''
        
    if weighted:
        tau, p_value = stats.weightedtau(true_ranks, pred_ranks)
    else:
        tau, p_value = stats.kendalltau(true_ranks, pred_ranks)

    return tau, p_value

def compute_spearmanr(true_ranks, pred_ranks, alternative='two-sided'):
    ''' Spearman rank-order correlation coefficient given two lists of ranks. '''

    corr, p_value = stats.spearmanr(true_ranks, pred_ranks, alternative=alternative)

    return corr, p_value

def compute_auc(pos_score, neg_score):
    ''' Compute ROC AUC score for arrays of positive and negative scores. '''
    
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
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Oct 28, 2021
@author: Simon Delarue <sdelarue@enst.fr>
'''

import numpy as np

def top_k(scores: np.ndarray, n: int = 1) -> np.ndarray:
    ''' Index of the n elements with highest value. 
        
        Parameters
        ----------
        scores: np.ndarray
            Array of scores.
        n: int
            Number of elements to return.
            
        Returns
        -------
        Array of n index. '''

    return np.argsort(-scores)[:n]

def MRR(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10, weight: bool = True) -> float:
    ''' Computes the Mean Reciprocal Rank between two arrays of scores restricted to k 
        elements. 
    
        Parameters
        ----------
        y_true: np.ndarray
            Array of ground truth scores.
        y_pred: np.ndarray
            Array of predicted scores.
        k: int (default = 10)
            Number of elements to consider.
        weight: bool (default = True)
            If True, normalize reciprocal rank with the importance of the element in `y_true`, i.e its rank.
            
        Returns
        -------
            Mean Reciprocal Score. '''

    top_node_true = top_k(y_true, k)
    node_pred = top_k(y_pred, len(y_pred))
    
    # Find reciprocal predicted ranks for top k elements in ground truth
    top_rank_pred = np.array([])
    for rank, node in enumerate(node_pred):
        if node in (top_node_true):
            '''recip_rank = 1 / (rank + 1)
            if weight:
                #rank_true = [i for i, val in enumerate(top_node_true) if val == node][0]
                rank_true = np.where([top_node_true == node])[1]
                weight = (k - rank_true) / k
                top_rank_pred = np.append(top_rank_pred, recip_rank * weight)
            else:
                top_rank_pred = np.append(top_rank_pred, recip_rank)'''
            rank_true = [i for i, val in enumerate(top_node_true) if val == node][0]
            rank_true = np.where([top_node_true == node])[1]
            recip_rank = 1 / (np.abs(rank - rank_true) + 1)
            top_rank_pred = np.append(top_rank_pred, recip_rank)

    # Compute MRR
    mrr = np.mean(top_rank_pred)
    
    return mrr

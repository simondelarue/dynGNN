# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from scipy import sparse
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import seaborn as sns

import dgl
import torch

def write_log(filename, text):
    with open(filename, 'a') as f:
        f.write(text)

def find_triplets(params, target_sum):
    triplets = []
    for val_a in params:
        for val_b in params:
            for val_g in params:
                if ((val_a + val_b + val_g) == target_sum):
                    triplets.append([val_a, val_b, val_g])
    return triplets

def sample_random_node(g, n):
    return np.random.choice(g.nodes().numpy(), n)

def sample_random_time(timerange, n):
    return np.random.choice(timerange, n)

def build_random_edges(src_list, dest_list, time_list):
    res = []
    for i, j, t in zip(src_list, dest_list, time_list):
        res.append((int(i), int(j), int(t)))
    return res

def duplicates(l1, l2):
    return list((set(l1).intersection(set(l2))))

def sample_non_neighbors(g, node, all_nodes):
    ''' Return a list of non neighbors of a specific node in a graph. '''
    
    # Neighbors nodes
    src_nodes = dgl.in_subgraph(g, int(node)).edges()[0]
    # Non neighbors nodes
    non_neighb_nodes = np.array(list(all_nodes - set(src_nodes.cpu().numpy())))
    
    return torch.from_numpy(non_neighb_nodes)

def sample_neighbors(g, node):
    ''' Return a list of neighbors of a specific node in a graph. '''
    
    # Neighbors nodes
    src_nodes = dgl.in_subgraph(g, int(node)).edges()[0]
    
    return src_nodes     


def make_edge_list(src, dest, t, mask):
    return [(u, v, t) for u, v, t in zip(src[mask].numpy(), dest[mask].numpy(), t[mask].numpy())]

def normalize_adj(A):
    
    A_arr = A.numpy()

    # Compute degrees of adjacency matrix
    deg = A_arr.dot(np.ones(A_arr.shape[1]))
    
    D = np.diag(deg)
    sparse_D = sparse.coo_matrix(D)
    data = sparse_D.data
    sparse_D.data = 1 / np.sqrt(data)
    D_norm = sparse_D.todense()
    
    # Normalize adjacency matrix
    A_norm = D_norm.dot(A_arr).T.dot(D_norm).T
    
    return torch.from_numpy(A_norm)

def compute_agg_features(g, timerange, add_self_edges=True):
    
    # Add edges between node and itself
    if add_self_edges:
        g.add_edges(g.nodes(), g.nodes())
    
    src, dest = g.edges()
    adj = coo_matrix((np.ones(len(src)), (src.numpy(), dest.numpy())))
    
    norm_mat = np.eye(adj.shape[0]) * (1 / len(timerange))
    norm_adj = torch.from_numpy(adj.dot(norm_mat))
    
    return norm_adj

def compute_agg_features_simplified(g, timerange, add_self_edges=True):
    
    # Add edges between node and itself
    if add_self_edges:
        g.add_edges(g.nodes(), g.nodes())
    
    src, dest = g.edges()
    adj = coo_matrix((np.ones(len(src)), (src.numpy(), dest.numpy())))
    
    return adj

def negative_sampling(g, timerange, list_pos_edges):

    list_neg_edges = []
    nb_pos_edges = len(list_pos_edges)
    nb_dup = nb_pos_edges
    # Create as much as negative edges than positive edges. Iterates until less than nb of positive edges/1000
    # are similar between negative and positive sampling.
    while (len(list_neg_edges) < nb_pos_edges) and (nb_dup > int(nb_pos_edges/1000)):
        nb_to_create = nb_pos_edges - len(list_neg_edges)
        list_neg_edges = build_random_edges(sample_random_node(g, nb_to_create),
                                            sample_random_node(g, nb_to_create),
                                            sample_random_time(timerange, nb_to_create))
        dup = duplicates(list_neg_edges, list_pos_edges)
        nb_dup = len(dup)
        if nb_dup > int(nb_pos_edges/1000):
            # Delete edges that appears in positive set
            list_neg_edges = [elem for elem in list_neg_edges if elem not in dup]
        else:
            break

    return list_neg_edges

"""def temporal_sampler(g, batch_size, timestep):
    ''' Returns a list of subgraph according to desired batch size. '''
    
    batches = []
    indexes = [] # returns list of index with 1 if batch-graph was returned, else 0
    
    batch_period = timestep * batch_size
    timerange = np.arange(int(g.edata['timestamp'].min()), int(g.edata['timestamp'].max()), batch_period)
    eids = np.arange(g.number_of_edges())
    
    for period in timerange:
    
        # Edges to remove
        rm_eids = eids[torch.logical_not(torch.logical_and(g.edata['timestamp'] >= period, 
                                                           g.edata['timestamp'] < (period + batch_period)))]
        
        batch_g = dgl.remove_edges(g, rm_eids) # also remove the feature attached to the edge
        
        # Later, use indexes to consider graph batch only if edges exist inside
        batches.append(batch_g)
        
        if batch_g.number_of_edges() != 0:
            indexes.append(True)
        else:
            indexes.append(False)
        
    return batches, indexes   """

def nb_edges_at_ts(df, start, nb_ts):
    ''' Returns the total number of edges in link stream, starting from timestep 'start', during 'nb_ts' number of timesteps. '''

    df_tmp = df[(df['t'] >= start)].copy()
    ts_k = df_tmp['t'].unique()[:nb_ts]
    df_filt = df_tmp[df_tmp['t'] <= ts_k[-1]]
    nb_edges = df_filt.shape[0]

    return nb_edges

def plot_history_loss(history, ax, label=None):
    if label is not None:
        ax.plot(range(len(history)), history, label=f'Loss - {label}')
        ax.legend()
    else:
        ax.plot(range(len(history)), history)
    #ax.set_ylim(0, 1)
    ax.set_title('Training loss', weight='bold')
    ax.set_xlabel('batches')
    ax.set_ylabel('Loss')

def plot_auc(res, ax, title, label=None):
    ax.plot(res['test_fpr'], res['test_tpr'], label=f"AUC={100*res['test_auc']:.3f}% {label}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.5)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(f'ROC AUC Curve - {title}', weight='bold')
    ax.legend();

def plot_classification_report(res, ax, title, label=None):
    sns.heatmap(pd.DataFrame(res).iloc[:-1, :].T, annot=True, ax=ax).set_title('Classification report')

def plot_result(history, ax, title, label, metric):
    if metric=='auc':
        plot_auc(history, ax, title, label)
    elif metric=='classification_report':
        plot_classification_report(history, ax, title, label)
    elif metric=='f1_score':
        plot_classification_report(history, ax, title, label)

def print_result(history, metric):
    if metric=='auc':
        print(f" ====> Test ROC AUC : {history['test_auc']:.4f}")
    elif metric=='classification_report':
        print(f" ====> Test Classification report : {history['test_classification_report']}")
    elif metric=='f1_score':
        print(f" ===> Test F1 score : {history['test_f1_score']:.4f}")

def save_figures(fig, path, name, ext='png'):
    if not os.path.isdir(path):
        os.mkdir(path)
    fig.savefig(f'{path}/{name}.{ext}')

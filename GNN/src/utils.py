# -*- coding: utf-8 -*-
import numpy as np
import torch
from scipy import sparse
import scipy.sparse as sp
import dgl
from dgl.data.utils import save_graphs
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

def compute_loss(pos_score, neg_score, device):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    
    # Compute auc
    scores = torch.cat([pos_score, neg_score]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().numpy()
    
    # Compute fpr and tpr
    fpr, tpr, _ = roc_curve(labels, scores)
    
    return roc_auc_score(labels, scores), fpr, tpr

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
    return int(np.random.choice(g.nodes().numpy(), n))

def sample_random_time(timerange, n, mask):
    return int(np.random.choice(timerange[mask], n))

def sample_non_neighbors(g, node, all_nodes):
    ''' Return a list of non neighbors of a specific node in a graph. '''
    
    # Neighbors nodes
    src_nodes = dgl.in_subgraph(g, int(node)).edges()[0]
    # Non neighbors nodes
    non_neighb_nodes = np.array(list(all_nodes - set(src_nodes.numpy())))
    
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
    
    return A_norm

def compute_batch_feature(g, timerange, add_self_edge=True):
    
    # Add edges between node and itself
    if add_self_edge:
        g.add_edges(g.nodes(), g.nodes())
    
    src, dest = g.edges()
    adj = np.zeros((g.number_of_nodes(), g.number_of_nodes()))
    for src_val, dest_val in zip(src, dest):
        adj[src_val, dest_val] += 1
    
    norm_mat = np.eye(adj.shape[0]) * (1 / len(timerange))
    norm_adj = adj.dot(norm_mat)
    
    return norm_adj

def train_test_split(g, val_size=0.15, test_size=0.15):

    print('\nSplitting graph ...')

    # Compute validation time and test time
    val_cut = 1 - val_size - test_size
    test_cut = 1 - test_size
    val_time, test_time = np.quantile(g.edata['timestamp'], [val_cut, test_cut])

    # Split edge set for training, validation and test
    # Edges are divided into 2 groups : positive (link in graph) and negative (no link in graph)

    source, dest = g.edges()
    timestamp = g.edata['timestamp']

    # Masks for datasets filtering
    train_mask = g.edata['timestamp'] <= val_time
    val_mask = torch.logical_and(g.edata['timestamp'] > val_time, g.edata['timestamp'] <= test_time)
    test_mask = g.edata['timestamp'] > test_time

    eids = np.arange(g.number_of_edges())
    val_nb_edge = len(source[val_mask])
    test_nb_edge = len(source[test_mask])
    train_nb_edge = len(eids) - test_nb_edge - val_nb_edge

    # -- Positive edges --
    # Postiive edges are used to create positive graphs according to splits

    train_pos_g = dgl.graph((source[train_mask], dest[train_mask]), num_nodes=g.number_of_nodes())
    val_pos_g = dgl.graph((source[val_mask], dest[val_mask]), num_nodes=g.number_of_nodes())
    test_pos_g = dgl.graph((source[test_mask], dest[test_mask]), num_nodes=g.number_of_nodes())
    train_pos_g.edata['timestamp'] = timestamp[train_mask]
    val_pos_g.edata['timestamp'] = timestamp[val_mask]
    test_pos_g.edata['timestamp'] = timestamp[test_mask]

    # -- Negative edges --
    # Negative edges are sampled randomly and used to create negative graphs according to splits

    edge_list_train = make_edge_list(source, dest, timestamp, train_mask)
    edge_list_val = make_edge_list(source, dest, timestamp, val_mask)
    edge_list_test = make_edge_list(source, dest, timestamp, test_mask)
    timerange = np.arange(int(g.edata['timestamp'].min()), int(g.edata['timestamp'].max()), 20)

    # Masks for negative edges according to splits
    train_mask_neg = timerange <= val_time
    val_mask_neg = np.logical_and(timerange > val_time, timerange <= test_time)
    test_mask_neg = timerange > test_time

    # Negative edges - Training set
    train_neg_edge = []
    while len(train_neg_edge) < train_nb_edge:
        random_edge = (sample_random_node(g, 1), sample_random_node(g, 1), sample_random_time(timerange, 1, train_mask_neg))
        if random_edge not in edge_list_train:
            train_neg_edge.append(random_edge)

    train_src_id, train_dest_id, train_t = zip(*train_neg_edge)
    train_neg_g = dgl.graph((train_src_id, train_dest_id), num_nodes=g.number_of_nodes())
    train_neg_g.edata['timestamp'] = torch.tensor(train_t)

    # Negative edges - Validation set
    val_neg_edge = []
    while len(val_neg_edge) < val_nb_edge:
        random_edge = (sample_random_node(g, 1), sample_random_node(g, 1), sample_random_time(timerange, 1, val_mask_neg))
        if random_edge not in edge_list_val:
            val_neg_edge.append(random_edge)

    val_src_id, val_dest_id, val_t = zip(*val_neg_edge)
    val_neg_g = dgl.graph((val_src_id, val_dest_id), num_nodes=g.number_of_nodes())
    val_neg_g.edata['timestamp'] = torch.tensor(val_t)

    # Negative edges - Test set
    test_neg_edge = []
    while len(test_neg_edge) < test_nb_edge:
        random_edge = (sample_random_node(g, 1), sample_random_node(g, 1), sample_random_time(timerange, 1, test_mask_neg))
        if random_edge not in edge_list_test:
            test_neg_edge.append(random_edge)

    test_src_id, test_dest_id, test_t = zip(*test_neg_edge)
    test_neg_g = dgl.graph((test_src_id, test_dest_id), num_nodes=g.number_of_nodes())
    test_neg_g.edata['timestamp'] = torch.tensor(test_t)

    # Build training graph (ie. graph without test and valid edges)
    train_g = dgl.remove_edges(g, eids[train_nb_edge:])

    # Save graphs
    graphs_to_save = [train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g]
    save_graphs(f'../preprocessed_data/data.bin', graphs_to_save)
    print('Done !')

    return train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g

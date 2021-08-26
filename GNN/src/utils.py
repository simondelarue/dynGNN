# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy.sparse as sp
import dgl

def sample_random_node(g, n):
    return int(np.random.choice(g.nodes().numpy(), n))

def sample_random_time(timerange, n, mask):
    return int(np.random.choice(timerange[mask], n))

def make_edge_list(src, dest, t, mask):
    return [(u, v, t) for u, v, t in zip(src[mask].numpy(), dest[mask].numpy(), t[mask].numpy())]

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

    print('Done !')

    return train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g

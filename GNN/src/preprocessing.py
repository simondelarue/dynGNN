# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
import dgl
import os
from utils import *


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

    # In addition to previous test set, we create a another test set which contains only edges
    # that appeared before in time (i.e in training set)
    seen_edges_tuple = [(int(src), int(dst)) for src, dst in zip(train_pos_g.edges()[0], train_pos_g.edges()[1])]
    test_edges_tuple = [(int(src), int(dst)) for src, dst in zip(test_pos_g.edges()[0], test_pos_g.edges()[1])]
    inter = list(set(test_edges_tuple).intersection(set(seen_edges_tuple)))
    test_eids = []
    for idx, tup in enumerate(test_edges_tuple):
        if tup in inter:
            test_eids.append(idx)
    test_seen_pos_g = dgl.edge_subgraph(test_pos_g, test_eids, preserve_nodes=True)
    test_seen_nb_edge = len(test_seen_pos_g.edges()[0])

    # -- Negative edges --
    # Negative edges are sampled randomly and used to create negative graphs according to splits

    edge_list_train = make_edge_list(source, dest, timestamp, train_mask)
    edge_list_val = make_edge_list(source, dest, timestamp, val_mask)
    edge_list_test = make_edge_list(source, dest, timestamp, test_mask)
    edge_list_seen_test = make_edge_list(test_seen_pos_g.edges()[0], test_seen_pos_g.edges()[1], test_seen_pos_g.edata['timestamp'], [True]*len(test_seen_pos_g.edata['timestamp']))
    timerange = np.arange(int(g.edata['timestamp'].min()), int(g.edata['timestamp'].max()), 20)

    # Masks for negative edges according to splits
    train_mask_neg = timerange <= val_time
    val_mask_neg = np.logical_and(timerange > val_time, timerange <= test_time)
    test_mask_neg = timerange > test_time

    # Negative edges - Training set
    train_neg_edge = negative_sampling(g, timerange[train_mask_neg], edge_list_train)

    train_src_id, train_dest_id, train_t = zip(*train_neg_edge)
    train_neg_g = dgl.graph((train_src_id, train_dest_id), num_nodes=g.number_of_nodes())
    train_neg_g.edata['timestamp'] = torch.tensor(train_t)

    # Negative edges - Validation set
    val_neg_edge = negative_sampling(g, timerange[val_mask_neg], edge_list_val)

    val_src_id, val_dest_id, val_t = zip(*val_neg_edge)
    val_neg_g = dgl.graph((val_src_id, val_dest_id), num_nodes=g.number_of_nodes())
    val_neg_g.edata['timestamp'] = torch.tensor(val_t)

    # Negative edges - Test set
    test_neg_edge = negative_sampling(g, timerange[test_mask_neg], edge_list_test)

    test_src_id, test_dest_id, test_t = zip(*test_neg_edge)
    test_neg_g = dgl.graph((test_src_id, test_dest_id), num_nodes=g.number_of_nodes())
    test_neg_g.edata['timestamp'] = torch.tensor(test_t)

    # Negative edges - Test seen set
    test_neg_seen_edge = negative_sampling(g, timerange[test_mask_neg], edge_list_seen_test)

    test_src_id, test_dest_id, test_t = zip(*test_neg_seen_edge)
    test_neg_seen_g = dgl.graph((test_src_id, test_dest_id), num_nodes=g.number_of_nodes())
    test_neg_seen_g.edata['timestamp'] = test_seen_pos_g.edata['timestamp']

    # Build training graph (ie. graph without test and valid edges)
    train_g = dgl.remove_edges(g, eids[train_nb_edge:])

    # Save graphs
    graphs_to_save = [train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g, test_neg_seen_g]
    save_graphs(f'../preprocessed_data/data.bin', graphs_to_save)
    print('Done !')

    return train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g, test_neg_seen_g


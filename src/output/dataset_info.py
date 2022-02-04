# -*- coding: utf-8 -*-

import argparse
from scipy import sparse
import numpy as np
import os
import torch
from utils import *
from gcn import *
from data_loader import DataLoader
from stream_graph import StreamGraph
import networkx as nx

def run(data, val_size, test_size, cache, batch_size, feat_struct, step_prediction, timestep, norm, emb_size, model_name, epochs, lr, metric, device, result_path):

    # ------ Load Data & preprocessing ------
    dl = DataLoader(data)

    # ------ Stream graph ------
    sg = StreamGraph(dl)
    g = sg.g

    # Split
    sg.train_test_split(val_size, test_size, timestep=timestep, neg_sampling=True)

    # Dataset info
    print('Number of nodes : ', g.number_of_nodes())
    print('Number of edges : ', g.number_of_edges())

    src, dest = g.edges()
    adj = sparse.coo_matrix((np.ones(len(src)), (src.numpy(), dest.numpy())), shape=(g.number_of_nodes(), g.number_of_nodes()))
    dense_adj = adj.todense()
    print(dense_adj[36, 144])
    print(dense_adj[144, 36])
    degrees = []
    weights = []
    agg_degrees = []
    W = len(sg.trange)*g.number_of_nodes()

    for node in range(dense_adj.shape[0]):
        degrees.append(dense_adj[node, :].sum() / len(sg.trange))
        filt_deg = dense_adj[node, :]
        agg_degrees.append(filt_deg[filt_deg != 0].shape[1])
        #weights.append(dense_adj[node, :].sum().sum() / (len(sg.trange)*g.number_of_nodes()))
        filt = dense_adj[node, :]
        weights.append(filt[filt != 0].shape[1] / W)

    res = 0
    for w, d in zip(weights, degrees):
        res += w * d

    # Density
    m = 2 * g.number_of_edges()
    m_tot = g.number_of_nodes() * (g.number_of_nodes() - 1)
    print(f'Dynamic graph density : {m / m_tot}')

    avg_agg_deg = (1 / g.number_of_nodes()) * sum(agg_degrees)
        
    #print(f'Average degree : {np.mean(degrees):.3f}')
    print(f'Average degree of dynamic graph : {res:.2e}')
    print(f'Average degree of agg graph : {avg_agg_deg:.2e}')

    
if __name__=='__main__':

    LOG_PATH = f'{os.getcwd()}/logs'
    #RESULT_PATH = f'{os.getcwd()}/results'
    RESULT_PATH = '/home/infres/sdelarue/node-embedding/GNN/results'

    parser = argparse.ArgumentParser('Preprocessing data')
    parser.add_argument('--data', type=str, help='Dataset name : \{SF2H\}', default='SF2H')
    parser.add_argument('--cache', type=str, help='Path for splitted graphs already cached', default=None)
    parser.add_argument('--feat_struct', type=str, help='Data structure : \{agg, time_tensor, temporal_edges\}', default='time_tensor')
    parser.add_argument('--step_prediction', type=str, help="If data structure is 'temporal_edges', either 'single' or 'multi' step predictions can be used.", default=None)
    parser.add_argument('--normalized', type=bool, help='If true, normalized adjacency is used', default=True)
    parser.add_argument('--timestep', type=int, default=20)
    parser.add_argument('--model', type=str, help='GCN model : \{GraphConv, GraphSage, GCNTime\}', default='GraphConv')
    parser.add_argument('--batch_size', type=int, help='If batch_size > 0, stream graph is splitted into batches.', default=0)
    parser.add_argument('--emb_size', type=int, help='Embedding size', default=20)
    parser.add_argument('--epochs', type=int, help='Number of epochs in training', default=1)
    parser.add_argument('--lr', type=float, help='Learning rate in for training', default=0.001)
    parser.add_argument('--metric', type=str, help='Evaluation metric : \{auc, f1_score, classification_report\}', default='auc')
    args = parser.parse_args()

    # ------ Parameters ------
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

    BATCH_SIZE = args.batch_size
    TIMESTEP = args.timestep 
    EMB_SIZE = args.emb_size
    MODEL = args.model
    EPOCHS = args.epochs
    LR = args.lr
    FEAT_STRUCT = args.feat_struct
    STEP_PREDICTION = args.step_prediction
    NORM = args.normalized
    METRIC = args.metric
    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    print(f'Device : {DEVICE}')
    
    # ------ Sanity check ------
    if MODEL=='GCNTime' and FEAT_STRUCT!='time_tensor':
        raise Exception("'GCNTime' model should only be filled with 'time_tensor' value for feature structure parameter.")

    # ------ Run model ------
    run(args.data,
        VAL_SIZE,
        TEST_SIZE,
        args.cache,
        BATCH_SIZE,
        FEAT_STRUCT,
        STEP_PREDICTION,
        TIMESTEP,
        NORM,
        EMB_SIZE,
        MODEL,
        EPOCHS,
        LR,
        METRIC,
        DEVICE,
        RESULT_PATH)
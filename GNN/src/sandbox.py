# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import os
import time
from timesteps import *
import itertools
import matplotlib.pyplot as plt
import dgl
from dgl.data.utils import load_graphs, save_graphs
import torch
from utils import *
from temporal_sampler import temporal_sampler
from predictor import DotPredictor
from gcn import *
from data_loader import DataLoader
from stream_graph import StreamGraph

def run(data, val_size, test_size, cache, batch_size, feat_struct, step_prediction, norm, emb_size, model_name, epochs, lr, metric, device, result_path):

    # ------ Load Data & preprocessing ------
    dl = DataLoader(data)

    # ------ Stream graph ------
    sg = StreamGraph(dl)
    g = sg.g

    # ------ Train test split ------
    start = time.time()
    if cache != None:
        print('\nUse cached splitted graphs')
        glist = list(load_graphs(f"{os.getcwd()}/{args.cache}/data.bin")[0])
        train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g, test_pos_seen_g, test_neg_seen_g = glist 
    else:
        sg.train_test_split(val_size, test_size, neg_sampling=True)
    end = time.time()
    print(f'Elapsed time : {end-start}s')

    # ------ Create batches ------
    if batch_size != 0:
        sg.create_batches(batch_size)


    # ------ Compute features ------
    # Features are computed accordingly to data structure and/or model.
    start = time.time()
    sg.compute_features(feat_struct, add_self_edges=True, normalized=norm)
    end = time.time()
    
    
    # Step link prediction ----------------
    # Link prediction on test set is evaluated for each timestep. In order to maintain the number of edge's distribution over time, 
    # we perform negative sampling on each snapshot in the test set
    val_pos_g_list, val_neg_g_list = step_linkpred_preprocessing(sg.val_pos_g, sg.trange_val, negative_sampling=True)
    print(f'Len val_pos_g_list : {len(val_pos_g_list)}')
    print(f'Len val_neg_g_list : {len(val_neg_g_list)}')
    
    # ------ Number of edges per batch ------
    '''# Training set
    sum_pos = 0
    for idx, g in enumerate(sg.train_pos_batches):
        sum_pos += g.number_of_edges()
    sum_neg = 0
    for idx, g in enumerate(sg.train_neg_batches):
        sum_neg += g.number_of_edges()
    print(f'Sum pos : {sum_pos} - sum neg : {sum_neg}')
    print(f'training pos g : {sg.train_pos_g.number_of_edges()}')
    print(f'training neg g : {sg.train_neg_g.number_of_edges()}')

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    width = 0.3
    plt.bar([i for i in range(len(sg.train_pos_batches))], [sg.train_pos_batches[i].number_of_edges() for i in range(len(sg.train_pos_batches))], label=f'positive edges (total={sum_pos})', width=width)
    plt.bar([i+width for i in range(len(sg.train_neg_batches))], [sg.train_neg_batches[i].number_of_edges() for i in range(len(sg.train_neg_batches))], label=f'negative edges (total={sum_neg})', width=width)
    plt.xlabel('batches')
    plt.ylabel('|E|')
    plt.legend()
    plt.title(f'Number of edges by batch in training dataset (total |E|={sum_pos + sum_neg})', weight='bold')
    save_figures(fig, f'{result_path}', f'{data}_number_of_edges_train')'''
    
    # Evaluation set
    sum_pos = 0
    for idx, g in enumerate(val_pos_g_list):
        sum_pos += g.number_of_edges()
    sum_neg = 0
    for idx, g in enumerate(val_neg_g_list):
        sum_neg += g.number_of_edges()
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    width = 0.3
    plt.bar([i for i in range(len(sg.trange_val))], [val_pos_g_list[i].number_of_edges() for i in range(len(sg.trange_val))], label=f'positive edges (total={sum_pos})', width=width)
    plt.bar([i+width for i in range(len(sg.trange_val))], [val_neg_g_list[i].number_of_edges() for i in range(len(sg.trange_val))], label=f'negative edges (total={sum_neg})', width=width)
    plt.xlabel('timestep')
    plt.ylabel('|E|')
    plt.legend()
    plt.title(f'Number of edges by timestep in evaluation dataset (total |E|={sum_pos + sum_neg})', weight='bold')
    save_figures(fig, f'{result_path}', f'{data}_number_of_edges_eval')

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
    TIMESTEP = 20 # SF2H dataset timestep
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
        NORM,
        EMB_SIZE,
        MODEL,
        EPOCHS,
        LR,
        METRIC,
        DEVICE,
        RESULT_PATH)
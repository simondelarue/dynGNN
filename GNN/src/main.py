# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
from dgl.data.utils import load_graphs
import torch

from preprocessing import temporal_graph
from utils import train_test_split, compute_batch_feature, normalize_adj, compute_loss, \
    write_log
from temporal_sampler import temporal_sampler
from predictor import DotPredictor
from gcn import GCNNeighb, GCNNonNeighb

parser = argparse.ArgumentParser('Preprocessing data')
parser.add_argument('--data', type=str, help='Dataset name : \{SF2H\}', default='SF2H')
parser.add_argument('--cache', type=str, help='Path for splitted graphs already cached', default='preprocessed_data')
parser.add_argument('--emb_size', type=int, help='Embedding size', default=20)
parser.add_argument('--epochs', type=int, help='Number of epochs in training', default=1)
parser.add_argument('--lr', type=float, help='Learning rate in for training', default=0.001)
args = parser.parse_args()

# ------ Parameters ------
VAL_SIZE = 0.15
TEST_SIZE = 0.15

BATCH_SIZE = 1
TIMESTEP = 20 # SF2H dataset timestep
EMB_SIZE = args.emb_size
EPOCHS = args.epochs
LR = args.lr
LOG_PATH = f'{os.getcwd()}/logs'


# ------ Dynamic graph ------
g = temporal_graph(args.data)

# ------ Train test split ------
if args.cache != None:
    print('\nUse cached splitted graphs')
    glist = list(load_graphs(f"{os.getcwd()}/{args.cache}/data.bin")[0])
    train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = glist 
else:
    train_g, train_pos_g, train_neg_g, \
        val_pos_g, val_neg_g, \
        test_pos_g, test_neg_g = train_test_split(g, VAL_SIZE, TEST_SIZE)


# ------ Sample training batches ------
print(f'\nSampling training batches of size {BATCH_SIZE} ...')
# Build graph batches
train_batches, indexes_pos = temporal_sampler(train_g, BATCH_SIZE, TIMESTEP)
train_neg_batches, _ = temporal_sampler(train_neg_g, BATCH_SIZE, TIMESTEP)

# Filter graphs only for the period where a positive graph is computed
train_batches = np.array(train_batches)[indexes_pos]
train_neg_batches = np.array(train_neg_batches)[indexes_pos]

print(f'# of pos train batch-graph : {len(train_batches)}')
print(f'# of neg train batch-graph : {len(train_neg_batches)}')
print('Done!')


# ------ Compute batch features ------
print('\nComputing batch features ...')
for train_batch_g in train_batches:
    train_batch_timerange = np.arange(int(train_batch_g.edata['timestamp'].min()), int(train_batch_g.edata['timestamp'].max()) + 20, 20)
    # Compute features 
    if train_batch_timerange[0] != 0:
        train_batch_feat = compute_batch_feature(train_batch_g, train_batch_timerange, add_self_edge=False) 
        train_batch_g.ndata['feat'] = torch.from_numpy(normalize_adj(torch.from_numpy(train_batch_feat)))
        
for train_neg_batch_g in train_neg_batches:
    train_neg_batch_timerange = np.arange(int(train_neg_batch_g.edata['timestamp'].min()), int(train_neg_batch_g.edata['timestamp'].max()) + 20, 20)
    # Compute features 
    if train_neg_batch_timerange[0] != 0:
        train_neg_batch_feat = compute_batch_feature(train_neg_batch_g, train_neg_batch_timerange, add_self_edge=False) 
        train_neg_batch_g.ndata['feat'] = torch.from_numpy(normalize_adj(torch.from_numpy(train_neg_batch_feat)))

print('Done!')

# ====== Graph Neural Networks ======

# 1. Neighbor's embedding GCN ------

logfile = 'log_GCN_Neighbors.txt'
with open(f'{LOG_PATH}/{logfile}', 'w') as f:
    f.write('EXECUTION LOG \n\n')

history_emb_N = [] 

# Model
neighb_model = GCNNeighb(train_batches[0].ndata['feat'].shape[1], EMB_SIZE)
optimizer_neighb = torch.optim.Adam(neighb_model.parameters(), lr=LR)
pred = DotPredictor()

# Training
print('\n Neighbors GCN training ...')
neighb_model.train(optimizer=optimizer_neighb,
                    pos_batches=train_batches,
                    neg_batches=train_neg_batches,
                    emb_size=EMB_SIZE,
                    predictor=pred,
                    loss=compute_loss,
                    epochs=EPOCHS)

# Test
print('\n Neighbors GCN test ...')
history_score, test_pos_score, test_neg_score = neighb_model.test(pred, test_pos_g, test_neg_g, metric='auc', return_all=True)
print(f'Done!')
print(f" ====> Test AUC : {history_score['test_auc']}")
write_log(f'{LOG_PATH}/{logfile}', f"Test AUC : {history_score['test_auc']}")

# 2. Non-Neighbor's embedding GCN ------

logfile = 'log_GCN_NNeighbors.txt'
with open(f'{LOG_PATH}/{logfile}', 'w') as f:
    f.write('EXECUTION LOG \n\n')

history_emb_NN = [] 

# Model
non_neighb_model = GCNNonNeighb(train_batches[0].ndata['feat'].shape[1], EMB_SIZE)
optimizer_non_neighb = torch.optim.Adam(non_neighb_model.parameters(), lr=LR)
pred = DotPredictor()

# Training
'''print('\n Non-Neighbors GCN training ...')
non_neighb_model.train(optimizer=optimizer_non_neighb,
                    pos_batches=train_batches,
                    neg_batches=train_neg_batches,
                    emb_size=EMB_SIZE,
                    predictor=pred,
                    loss=compute_loss,
                    epochs=EPOCHS)

# Test
history_score, test_pos_score, test_neg_score = non_neighb_model.test(pred, test_pos_g, test_neg_g, metric='auc', return_all=True)
print(f'Done!')
print(f" ====> Test AUC : {history_score['test_auc']}")
write_log(f'{LOG_PATH}/{logfile}', f"Test AUC : {history_score['test_auc']}")'''

# 3. Full GCN : Neighbor's GCN + Non-Neighbor's GCN + Previous timestep embedding GCN ------
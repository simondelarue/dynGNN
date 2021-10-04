# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import os
import time
import itertools
from dgl.data.utils import load_graphs
import torch

from preprocessing import temporal_graph
from utils import train_test_split, compute_batch_feature, normalize_adj, compute_loss, \
    write_log
from temporal_sampler import temporal_sampler
from predictor import DotPredictor
from gcn import GCNNeighb, GCNNonNeighb, GCNModel_time

LOG_PATH = f'{os.getcwd()}/logs'
logfile = 'log_GCN_time_cuda.txt'
with open(f'{LOG_PATH}/{logfile}', 'w') as f:
    f.write('EXECUTION LOG \n\n')

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
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Device information
print(f'--- Using Device : {DEVICE}')
print(torch.cuda.get_device_name(0))

df_preproc = pd.read_pickle(f'{os.getcwd()}/preprocessed_data/SF2H.pkl')


# ------ Dynamic graph ------
g = temporal_graph(args.data)

# ------ Train test split ------
start = time.time()
if args.cache != None:
    print('\nUse cached splitted graphs')
    glist = list(load_graphs(f"{os.getcwd()}/{args.cache}/data.bin")[0])
    train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = glist 
else:
    train_g, train_pos_g, train_neg_g, \
        val_pos_g, val_neg_g, \
        test_pos_g, test_neg_g = train_test_split(g, VAL_SIZE, TEST_SIZE)

end = time.time()
print(f'Elapsed time : {end-start}s')
write_log(f'{LOG_PATH}/{logfile}', f"\nElapsed time : {end-start}")

# ------ Build Adjacency matrix ------

print('\nBuild Adjacency Tensor with time information ...')
# Training dataframe
max_train_time = int(train_pos_g.edata['timestamp'].max())
df_train = df_preproc[df_preproc['t']<=max_train_time]
train_timerange = list(np.arange(int(train_pos_g.edata['timestamp'].min()), int(train_pos_g.edata['timestamp'].max()) + 20, 20))
train_timerange_idx = [i for i in range(len(train_timerange))]
adj = torch.zeros(len(train_pos_g.nodes()), len(train_pos_g.nodes()), len(train_timerange_idx))

# Fill a tensor of size (nb nodes, nb nodes, timerange) with 1 if interaction exists, else 0
for node_src, node_dest, time in zip(df_train['src'], df_train['dest'], df_train['t']):
    time_index = train_timerange.index(time)
    adj[node_src, node_dest, time_index] = 1

adj_self_edges = adj.clone()
# Add self edges
for node in train_pos_g.nodes():
    adj_self_edges[node, node, :] = torch.ones(len(train_timerange_idx))

# Normalize adjacency
#norm_adj = normalize_adj(adj_self_edges)

# ------ Add features to g ------
train_g.ndata['feat'] = adj_self_edges
print('Done!')

# ====== Graph Neural Networks ======

# ------ Training ------
hist_tot = {}

# Initialize model
model = GCNModel_time(train_g.ndata['feat'].shape[0], EMB_SIZE, train_g.ndata['feat'].shape[2]).to(DEVICE)
pred = DotPredictor()

# Optimizer
optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=LR)

# Training
#start = time.time()
print('\nFull GCN training ...')
model.train(optimizer=optimizer,
                    train_g=train_g,
                    train_pos_g=train_pos_g,
                    train_neg_g=train_neg_g,
                    predictor=pred,
                    loss=compute_loss,
                    device=DEVICE,
                    epochs=EPOCHS)

print('\n GCN test ...')
history_score, test_pos_score, test_neg_score = model.test(pred, test_pos_g, test_neg_g, metric='auc', return_all=True)
print(f'Done!')
print(f" ====> Test AUC : {history_score['test_auc']:.4f}")
write_log(f'{LOG_PATH}/{logfile}', f"Test AUC : {history_score['test_auc']}")

# Save results
logfile_history = 'history_time'
np.save(f'{LOG_PATH}/{logfile_history}.npy', history_score)
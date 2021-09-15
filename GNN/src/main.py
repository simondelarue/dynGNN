# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import time
from dgl.data.utils import load_graphs, save_graphs
import torch

from preprocessing import temporal_graph
from utils import train_test_split, compute_batch_feature, normalize_adj, compute_loss, \
    write_log, find_triplets
from temporal_sampler import temporal_sampler
from predictor import DotPredictor
from gcn import GCNNeighb, GCNNonNeighb, GCNModelFull

LOG_PATH = f'{os.getcwd()}/logs'
logfile = 'log_GCN_cuda.txt'
with open(f'{LOG_PATH}/{logfile}', 'w') as f:
    f.write('EXECUTION LOG \n\n')

parser = argparse.ArgumentParser('Preprocessing data')
parser.add_argument('--data', type=str, help='Dataset name : \{SF2H\}', default='SF2H')
parser.add_argument('--cache', type=str, help='Path for splitted graphs already cached', default='preprocessed_data')
parser.add_argument('--batches', type=str, help='1 if cached batches are available, else 0', default='0')
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

# ------ Sample training batches ------
start = time.time()
print(f'\nSampling training batches of size {BATCH_SIZE} ...')
# Build graph batches
if args.batches != str(0):
    print('Use cached batches')
    train_batches = list(load_graphs(f"{os.getcwd()}/{args.cache}/batches_pos/data.bin")[0])
    train_neg_batches = list(load_graphs(f"{os.getcwd()}/{args.cache}/batches_neg/data.bin")[0])
else:
    train_batches, indexes_pos = temporal_sampler(train_g, BATCH_SIZE, TIMESTEP, DEVICE)
    train_neg_batches, _ = temporal_sampler(train_neg_g, BATCH_SIZE, TIMESTEP, DEVICE)
   # Filter graphs only for the period where a positive graph is computed
    train_batches = np.array(train_batches)[indexes_pos]
    train_neg_batches = np.array(train_neg_batches)[indexes_pos]

end = time.time()
print(f'Elapsed time : {end-start}s')
write_log(f'{LOG_PATH}/{logfile}', f"\nElapsed time : {end-start}")

print(f'# of pos train batch-graph : {len(train_batches)}')
print(f'# of neg train batch-graph : {len(train_neg_batches)}')
print('Done!')


# ------ Compute batch features ------

if args.batches == str(0):
    start = time.time()
    print('\nComputing batch features ...')

    for train_batch_g in train_batches:
        train_batch_timerange = np.arange(int(train_batch_g.edata['timestamp'].min()), int(train_batch_g.edata['timestamp'].max()) + 20, 20)
        # Compute features 
        if train_batch_timerange[0] != 0:
            train_batch_feat = compute_batch_feature(train_batch_g, train_batch_timerange, add_self_edge=False) 
            train_batch_g.ndata['feat'] = torch.from_numpy(normalize_adj(torch.from_numpy(train_batch_feat))) #.to(DEVICE)
            
    for train_neg_batch_g in train_neg_batches:
        train_neg_batch_timerange = np.arange(int(train_neg_batch_g.edata['timestamp'].min()), int(train_neg_batch_g.edata['timestamp'].max()) + 20, 20)
        # Compute features 
        if train_neg_batch_timerange[0] != 0:
            train_neg_batch_feat = compute_batch_feature(train_neg_batch_g, train_neg_batch_timerange, add_self_edge=False) 
            train_neg_batch_g.ndata['feat'] = torch.from_numpy(normalize_adj(torch.from_numpy(train_neg_batch_feat))) #.to(DEVICE)

    # Save graphs
    save_graphs(f'{os.getcwd()}/{args.cache}/batches_pos/data.bin', list(train_batches))
    save_graphs(f'{os.getcwd()}/{args.cache}/batches_neg/data.bin', list(train_neg_batches))

    end = time.time()
    print(f'Elapsed time : {end-start}s')
    write_log(f'{LOG_PATH}/{logfile}', f"\nElapsed time : {end-start}")
    print('Done!')


# ====== Graph Neural Networks ======

# 1. Neighbor's embedding GCN ------

history_emb_N = [] 

# Model
neighb_model = GCNNeighb(train_batches[0].ndata['feat'].shape[1], EMB_SIZE).to(DEVICE)
optimizer_neighb = torch.optim.Adam(neighb_model.parameters(), lr=LR)
pred = DotPredictor()

# Training
start = time.time()
print('\n Neighbors GCN training ...')
neighb_model.train(optimizer=optimizer_neighb,
                    pos_batches=train_batches,
                    neg_batches=train_neg_batches,
                    emb_size=EMB_SIZE,
                    predictor=pred,
                    loss=compute_loss,
                    device=DEVICE,
                    epochs=EPOCHS)
end = time.time()
print(f'Elapsed time : {end-start}s')

# Test
start = time.time()
print('Neighbors GCN test ...')
test_pos_g = test_pos_g.to(DEVICE)
test_neg_g = test_neg_g.to(DEVICE)
history_score, test_pos_score, test_neg_score = neighb_model.test(pred, test_pos_g, test_neg_g, metric='auc', return_all=True)
print(f'Done!')
print(f" ====> Test AUC : {history_score['test_auc']:.4f}")
write_log(f'{LOG_PATH}/{logfile}', f"Test AUC : {history_score['test_auc']}")

end = time.time()
print(f'Elapsed time : {end-start}s')
write_log(f'{LOG_PATH}/{logfile}', f"\nElapsed time : {end-start}")

# 2. Non-Neighbor's embedding GCN ------

#logfile = 'log_GCN_NNeighbors.txt'
#with open(f'{LOG_PATH}/{logfile}', 'w') as f:
#    f.write('EXECUTION LOG \n\n')

history_emb_NN = [] 

# Model
non_neighb_model = GCNNonNeighb(train_batches[0].ndata['feat'].shape[1], EMB_SIZE).to('cpu')
optimizer_non_neighb = torch.optim.Adam(non_neighb_model.parameters(), lr=LR)
pred = DotPredictor()
test_pos_g = test_pos_g.to('cpu')
test_neg_g = test_neg_g.to('cpu')

# Training
print('\n Non-Neighbors GCN training ...')
start = time.time()
non_neighb_model.train(optimizer=optimizer_non_neighb,
                    pos_batches=train_batches,
                    neg_batches=train_neg_batches,
                    emb_size=EMB_SIZE,
                    predictor=pred,
                    loss=compute_loss,
                    device='cpu',
                    epochs=EPOCHS)

# Test
print('Non-Neighbors GCN test ...')
history_score, test_pos_score, test_neg_score = non_neighb_model.test(pred, test_pos_g, test_neg_g, metric='auc', return_all=True)
print(f'Done!')
print(f" ====> Test AUC : {history_score['test_auc']:.4f}")
write_log(f'{LOG_PATH}/{logfile}', f"Test AUC : {history_score['test_auc']}")

end = time.time()
print(f'Elapsed time : {end-start}s')
write_log(f'{LOG_PATH}/{logfile}', f"\nElapsed time : {end-start}")

# 3. Full GCN : Neighbor's GCN + Non-Neighbor's GCN + Previous timestep embedding GCN ------

print('\n Full GCN training ...')
start = time.time()

# Previous Embeddings
emb_prev = torch.rand(train_batches[0].ndata['feat'].shape[0], 20, requires_grad=False) 
emb_N_saved_cp = neighb_model.history_train_['train_emb'].copy()
emb_NN_saved_cp = non_neighb_model.history_train_['train_emb'].copy()

# Parameters
alphas = np.arange(0, 1.1, 0.1)
betas = np.arange(0, 1.1, 0.1)
gammas = np.arange(0, 1.1, 0.1)
triplets = find_triplets(alphas, 1)
test_pos_g = test_pos_g.to(DEVICE)
test_neg_g = test_neg_g.to(DEVICE)

history_full = {}

# Training
for num_triplet, triplet in enumerate(triplets):

    alpha, beta, gamma = triplet
    print(f'Alpha : {alpha} - Beta : {beta}, gamma : {gamma}')
    print('Training ...')

    # Model
    full_model = GCNModelFull(train_batches[0].ndata['feat'].shape[1], EMB_SIZE).to(DEVICE)
    optimizer_full = torch.optim.Adam(full_model.parameters(), lr=LR)
    pred = DotPredictor()

    full_model.train(optimizer=optimizer_full,
                        pos_batches=train_batches,
                        neg_batches=train_neg_batches,
                        emb_size=EMB_SIZE,
                        predictor=pred,
                        loss=compute_loss,
                        device=DEVICE,
                        epochs=EPOCHS,
                        emb_prev=emb_prev,
                        emb_neighbors=emb_N_saved_cp,
                        emb_nneighbors=emb_NN_saved_cp,
                        alpha=alpha, beta=beta, gamma=gamma)

    # -- Test --
    print('Full GCN test ...')
    history_score, test_pos_score, test_neg_score = full_model.test(pred, test_pos_g, test_neg_g, metric='auc', return_all=True)
    print(f" ====> Test AUC : {history_score['test_auc']:.4f}")

    history_full[num_triplet] = history_score

# Save results
logfile = 'results_1_step'
np.save(f'{LOG_PATH}/{logfile}.npy', history_full)
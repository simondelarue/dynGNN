# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import os
import time
import itertools
import matplotlib.pyplot as plt
import dgl
from dgl.data.utils import load_graphs, save_graphs
import torch
from utils import *
from timesteps import *
from temporal_sampler import temporal_sampler
from predictor import DotPredictor, CosinePredictor
from gcn import *
from data_loader import DataLoader
from stream_graph import StreamGraph

def run(data, val_size, test_size, cache, batch_size, feat_struct, step_prediction, timestep, norm, emb_size, model_name, epochs, lr, metric, device, result_path, dup_edges):

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
        sg.train_test_split(val_size, test_size, timestep=timestep, neg_sampling=True)
    end = time.time()
    print(f'Elapsed time : {end-start}s')

    # ------ Deduplicate edges in training graphs ------
    print('Duplicate edges : ', dup_edges)
    if dup_edges == 'False':
        sg.train_g = dgl.to_simple(sg.train_g, copy_ndata=True, copy_edata=True)
        sg.train_pos_g = dgl.to_simple(sg.train_pos_g, copy_ndata=True, copy_edata=True)
        sg.train_neg_g = dgl.to_simple(sg.train_neg_g, copy_ndata=True, copy_edata=True)
    
    # ------ Create batches ------
    if batch_size != 0:
        sg.create_batches(batch_size)
        
    # ------ Compute features ------
    # Features are computed accordingly to data structure and/or model.
    start = time.time()
    sg.compute_features(feat_struct, add_self_edges=True, normalized=norm)
    end = time.time()
    print(f'Elapsed time : {end-start}s')
    
    # Step link prediction ----------------
    # Link prediction on test set is evaluated for each timestep. In order to maintain the number of edge's distribution over time, 
    # we perform negative sampling on each snapshot in the test set
    val_pos_g_list, val_neg_g_list = step_linkpred_preprocessing(sg.val_pos_g, sg.trange_val, negative_sampling=True)



    # ====== Graph Neural Networks ======

    # Graphs are converted to undirected before message passing algorithm
    if feat_struct != 'temporal_edges':
        sg.directed2undirected(copy_ndata=True, copy_edata=True)

    
    # Initialize model
    if model_name == 'GCNTime':
        model = GCNModelTime(sg.train_g.ndata['feat'].shape[0], emb_size, sg.train_g.ndata['feat'].shape[2]).to(device)
        models = [model]
    elif model_name == 'GraphConv':
        model = GCNGraphConv(sg.train_g.ndata['feat'].shape[0], emb_size).to(device)
        models = [model]
    elif model_name == 'GraphSage':
        model = GraphSAGE(sg.train_g.ndata['feat'].shape[0], emb_size).to(device)
        models = [model]
    elif model_name == 'GCN_lc':
        model_N = GCNNeighb(sg.train_pos_batches[0].ndata['feat'].shape[0], emb_size).to(device)
        model_NN = GCNNonNeighb(sg.train_pos_batches[0].ndata['feat'].shape[0], emb_size).to(device)
        model_full = GCNModelFull(sg.train_pos_batches[0].ndata['feat'].shape[0], emb_size).to(device)
        models = [model_N, model_NN, model_full]

    # Predictor
    pred = DotPredictor()
    #pred = CosinePredictor()

    # Train model
    kwargs = {'train_pos_g': sg.train_pos_g, 'train_neg_g': sg.train_neg_g}

    for i, model in enumerate(models):
        
        if model_name != 'GCN_lc':
            # Optimizer
            optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=LR)

            # Training
            start = time.time()
            print('\nGCN training ...')
            print(f'Training timerange length : {len(sg.trange_train)}')
            model.train(optimizer=optimizer,
                        predictor=pred,
                        loss=compute_loss,
                        device=device,
                        epochs=epochs,
                        **kwargs)
            end = time.time()
            print(f'Elapsed time : {end-start}s')

        elif i in [0, 1] and model_name == 'GCN_lc':
            kwargs = {'train_pos_batches': sg.train_pos_batches,
                      'train_neg_batches': sg.train_neg_batches,
                      'emb_size': emb_size}
            # Optimizer
            optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=LR)

            # Training
            start = time.time()
            print('\nGCN training ...')
            model.train(optimizer=optimizer,
                        predictor=pred,
                        loss=compute_loss,
                        device=device,
                        epochs=epochs,
                        **kwargs)
            end = time.time()
            print(f'Elapsed time : {end-start}s')

        elif i==2:
            alphas = np.arange(0, 1.25, 0.25)
            betas = np.arange(0, 1.25, 0.25)
            gammas = np.arange(0, 1.25, 0.25)
            triplets = find_triplets(alphas, 1)
            trained_models = []

            kwargs = {'train_pos_batches': sg.train_pos_batches,
                      'train_neg_batches': sg.train_neg_batches,
                        'emb_size': emb_size, 
                        'emb_prev': torch.rand(sg.train_pos_batches[0].ndata['feat'].shape[0], emb_size, requires_grad=False), 
                        'emb_neighbors': models[0].history_train_['train_emb'].copy(), 
                        'emb_nneighbors': models[1].history_train_['train_emb'].copy()}
            
            for num_triplet, triplet in enumerate(triplets):
                kwargs['alpha'], kwargs['beta'], kwargs['gamma'] = triplet

                # Model
                model_full = GCNModelFull(sg.train_pos_batches[0].ndata['feat'].shape[0], emb_size).to(device)
                # Optimizer
                optimizer = torch.optim.Adam(itertools.chain(model_full.parameters()), lr=LR)

                # Training
                start = time.time()
                print(f'\nGCN training (linear combination parameters : {triplet}) ...')
                model_full.train(optimizer=optimizer,
                            predictor=pred,
                            loss=compute_loss,
                            device=device,
                            epochs=epochs,
                            **kwargs)
                end = time.time()
                trained_models.append(model_full)
                print(f'Elapsed time : {end-start}s')
                
        
    # Evaluation
    print('\nGCN Eval ...')
    print(f'Evaluation timerange length : {len(sg.trange_val)}')
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    df_tot = pd.DataFrame()

    if model_name == 'GCN_lc':
        models = trained_models

    for idx, trained_model in enumerate(models):
        
        k_indexes = None
        if feat_struct=='temporal_edges':
            k_indexes = sg.last_k_emb_idx

        # Full evaluation dataset
        print(f'VAL POS G : {sg.val_pos_g}')
        print(f'VAL NEG G : {sg.val_neg_g}')
        history_score, val_pos_score, val_neg_score = trained_model.test(pred, 
                                                            sg.val_pos_g, 
                                                            sg.val_neg_g, 
                                                            metric=metric, 
                                                            feat_struct=feat_struct, 
                                                            step_prediction=step_prediction,
                                                            k_indexes=k_indexes,
                                                            return_all=True)

        if len(models) > 1:
            label = f'{triplets[idx]}'
        else:
            label = f'{model_name}'

        df_tmp = pd.DataFrame([[label, history_score['test_auc'], sg.val_pos_g.number_of_edges(), sg.val_neg_g.number_of_edges]], 
                                columns=['model', 'score', 'number_of_edges_pos', 'number_of_edges_neg'])
        df_tot = pd.concat([df_tot, df_tmp])

        # Step evaluation dataset
        '''for val_pos_g, val_neg_g, t in zip(val_pos_g_list, val_neg_g_list, sg.trange_val):
            if val_pos_g.number_of_edges() > 0 and val_neg_g.number_of_edges() > 0:
                history_score, val_pos_score, val_neg_score = trained_model.test(pred, 
                                                                    val_pos_g, 
                                                                    val_neg_g, 
                                                                    metric=metric, 
                                                                    feat_struct=feat_struct, 
                                                                    step_prediction=step_prediction,
                                                                    k_indexes=k_indexes,
                                                                    return_all=True)
                #print(f'Done !')
                #print_result(history_score, metric)

                # Plot Results
                #hist_train_loss = [float(x) for x in trained_model.history_train_['train_loss']]
                if len(models) > 1:
                    label = f'{triplets[idx]}'
                else:
                    label = f'{model_name}'

                df_tmp = pd.DataFrame([[label, history_score['test_auc'], t, val_pos_g.number_of_edges(), val_neg_g.number_of_edges]], 
                                        columns=['model', 'score', 'timestep', 'number_of_edges_pos', 'number_of_edges_neg'])
                df_tot = pd.concat([df_tot, df_tmp])

                #plot_history_loss(hist_train_loss, ax=ax[0], label=label)
                #ax[0].set_xlabel('epochs')       
                #plot_result(history_score, ax=ax[1], title='Eval set - unseen nodes', label=label, metric=metric)

    print('Done !')'''

    # Save results
    res_path = f'{result_path}/{data}/{feat_struct}'
    if feat_struct == 'temporal_edges':
        res_filename = f'{data}_GCN_{model_name}_{feat_struct}_eval_{metric}_{step_prediction}'
    else:
        res_filename = f'{data}_GCN_{model_name}_{feat_struct}_eval_{metric}'

    df_tot.to_pickle(f'{res_path}/{res_filename}.pkl', protocol=3)
    print(f'Results saved in {res_path}/{res_filename}.pkl')
    print(df_tot.shape)

    
    # Save results
    '''res_path = f'{result_path}/{data}/{feat_struct}'
    if feat_struct == 'temporal_edges':
        res_filename = f'{data}_GCN_{model_name}_{feat_struct}_unseen_eval_{metric}_{step_prediction}'
    else:
        res_filename = f'{data}_GCN_{model_name}_{feat_struct}_unseen_eval_{metric}'

    save_figures(fig, res_path, res_filename)'''
    
    # Test
    '''print('\n GCN test ...')
    history_score, test_pos_score, test_neg_score = model.test(pred, sg.test_pos_g, sg.test_neg_g, metric=metric, return_all=True)
    print(f'Done !')
    print_result(history_score, metric)
    # Plot Results
    hist_train_loss = [float(x) for x in model.history_train_['train_loss']]
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    plot_history_loss(hist_train_loss, ax=ax[0], label=f'{model_name}')
    ax[0].set_xlabel('epochs')
    plot_result(history_score, ax=ax[1], title='Test set - unseen nodes', label=f'{model_name}', metric=metric)
    fig.savefig(f'{result_path}/{data}_GCN_{model_name}_unseen_test_{metric}.png')

    print('\n GCN test seen ...')
    history_score, test_pos_score, test_neg_score = model.test(pred, sg.test_pos_seen_g, sg.test_neg_seen_g, metric=metric, return_all=True)
    print(f'Done !')
    print_result(history_score, metric)
    # Plot Results
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    plot_history_loss(hist_train_loss, ax=ax[0], label=f'{model_name}')
    ax[0].set_xlabel('epochs')
    plot_result(history_score, ax=ax[1], title='Test set - unseen nodes', label=f'{model_name}', metric=metric)
    fig.savefig(f'{result_path}/{data}_GCN_{model_name}_seen_test_{metric}.png')'''

# TODO

# - Ajout du jeu de données AS
# - Pdag simplifié
# - Utilisation de la métrique de score mAP (average_precision_score() from scikit-learn)
# - Tâche de ML sous-jacente : Classification

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
    parser.add_argument('--timestep', type=int, help='Finest granularity in temporal data.', default=20)
    parser.add_argument('--emb_size', type=int, help='Embedding size', default=20)
    parser.add_argument('--epochs', type=int, help='Number of epochs in training', default=1)
    parser.add_argument('--lr', type=float, help='Learning rate in for training', default=0.001)
    parser.add_argument('--metric', type=str, help='Evaluation metric : \{auc, f1_score, classification_report\}', default='auc')
    parser.add_argument('--duplicate_edges', type=str, help='If true, allows duplicate edges in training graphs', default='True')
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
    DUP_EDGES = args.duplicate_edges

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
        RESULT_PATH,
        DUP_EDGES)


# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from scipy import stats
import os
import time
import itertools
import matplotlib.pyplot as plt

import dgl
from dgl.data.utils import load_graphs
import torch

from utils import *
from loss import *
from timesteps import *
from predictor import *
from gcn import *
from metrics import *
from data_loader import DataLoader
from stream_graph import StreamGraph

def run(data, val_size, test_size, cache, batch_size, feat_struct, step_prediction, timestep, norm, emb_size, model_name, \
        epochs, lr, metric, device, result_path, model_path, dup_edges, test_agg, predictor, loss_func, shuffle_test):

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
        sg.train_test_split(val_size, test_size, timestep=timestep, neg_sampling=True, metric=metric)
    end = time.time()
    print(f'Elapsed time : {end-start}s')

    
    # ====== Baseline models ======
    if model_name.startswith('baseline'):
        max_train = sg.edata_train[-1].item()
        max_val = sg.edata_val[-1].item()
        df_train = sg.data_df[sg.data_df['t']<=max_train].copy()
        df_val = sg.data_df[(sg.data_df['t']>max_train) & (sg.data_df['t']<=max_val)].copy()

        # Reindex times
        df_train['t_idx'] = stats.rankdata(df_train['t'], 'dense')
        df_val['t_idx'] = stats.rankdata(df_val['t'], 'dense')
        nodes = set(df_train['src'].unique()).union(df_train['dest'].unique())

        # Node embeddings: average rank over time
        embedds = {}
        for node in nodes:
            df_train_tmp = df_train[(df_train['src']==node) | (df_train['dest']==node)]
            embedds[node] = df_train_tmp['t_idx'].mean()

        # Compute node embedding according to average neighbors ages. This baseline method
        # allows to add structural information (based on neighborhood) in addition to temporal elements
        if model_name == 'baseline_neighbors':
            embedds_neighbs = {}
            for node in nodes:
                neighbs_emb = []
                df_train_tmp = df_train[(df_train['src']==node) | (df_train['dest']==node)]
                # Compute embeddings for neighbors
                for n in set(df_train_tmp['src'].unique()).union(set(df_train_tmp['dest'].unique())):
                    if n != node:
                        neighbs_emb.append(embedds.get(n))
                
                # Stack embeddings of node and its neighbors embeddings and compute mean
                neighbs_emb.append(embedds[node])
                embedds_neighbs[node] = np.mean(neighbs_emb)

            embedds = embedds_neighbs

        avg_embed = np.mean(np.array(list(embedds.values())))

        # Inter-contact time: embedding of node is computed according to its average inter-contact time with other nodes
        # Higher values of embeddings reflect sparse interractions over time, lower values reflect dense interactions.
        if model_name == 'baseline_inter_contact':
            embedds_inter_contact = {}
            for node in nodes:
                df_train_tmp = df_train[(df_train['src']==node) | (df_train['dest']==node)]
                if df_train_tmp.shape[0] > 1:
                    avg_inter_contact_time = (df_train_tmp['t'].values[1:] - df_train_tmp['t'].values[:-1]).mean()
                else:
                    avg_inter_contact_time = df_train_tmp['t'].values[0]
                embedds_inter_contact[node] = avg_inter_contact_time

            embedds = embedds_inter_contact

        # True ranks
        src, dest, ranks, dup_mask = sg.rank_edges(sg.data_df, sg.trange_val, metric=metric, timestep=timestep)
        if shuffle_test == 'True':
            ranks = shuffle(ranks)

        # Predicted ranks on test 
        l_scores, l_scores_bis = [], []
        for s, d in zip(src, dest):
            # Compute link score according to node embeddings. If node embedding does not exists, use overall average node embedding
            embed_s = embedds.get(s, avg_embed)
            embed_d = embedds.get(d, avg_embed)
            l_score = (2 * embed_s * embed_d) / (embed_s + embed_d) # Harmonic mean
            l_score_bis = (embed_s + embed_d) / 2
            l_scores.append(l_score)
            #l_scores_bis.append(l_score_bis)
        pred_ranks = np.argsort(l_scores)
        #pred_ranks = np.argsort(l_scores_bis)

        # Compute metric
        if metric.startswith('wkendall'):
            tau, _ = compute_kendall(ranks, pred_ranks, weighted=True)
        elif metric.startswith('kendall'):
            tau, p_value = compute_kendall(ranks, pred_ranks, weighted=False)
        elif metric.startswith('spearmanr'):
            rho, p_value = compute_spearmanr(ranks, pred_ranks)

        df_tot = pd.DataFrame([[model_name, tau, 0, sg.val_pos_g.number_of_edges(), sg.val_neg_g.number_of_edges(), None,
                                None, None, None]], 
                                columns=['model', 'score', 'timestep', 'number_of_edges_pos', 'number_of_edges_neg', 'test_agg', \
                                                'duplicate_edges', 'predictor', 'loss_func'])
        print(f'Metric : {metric} = {tau}')

    
    # ====== Custom models ======
    else:

        # ------ Deduplicate edges in training graphs ------
        print('Duplicate edges : ', dup_edges)
        if dup_edges == 'False':
            sg.train_g = dgl.to_simple(sg.train_g, copy_ndata=True, copy_edata=True)
            sg.train_pos_g = dgl.to_simple(sg.train_pos_g, copy_ndata=True, copy_edata=True)
            sg.train_neg_g = dgl.to_simple(sg.train_neg_g, copy_ndata=True, copy_edata=True)
        
        # ------ Create batches ------
        if batch_size != 0:
            sg.create_batches(batch_size, timestep)
            
        # ------ Compute features ------
        # Features are computed accordingly to data structure and/or model.
        start = time.time()
        sg.compute_features(feat_struct, add_self_edges=True, normalized=norm, timestep=timestep)
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
        elif model_name == 'DTFT':
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
        pred = pred_factory(predictor)

        # Loss
        loss = loss_factory(loss_func).compute

        # Saving information
        saved_model_path = f'{model_path}/{data}/{feat_struct}'
        model_filename = f'{data}_{model_name}_{feat_struct}_{dup_edges}_{predictor}_{loss_func}'


        # ------ Training ------

        # Train model only if necessary. In other words, if combination of parameters have already been used for training,
        # use existing corresponding pre-trained model.
        if not os.path.isfile(f'{saved_model_path}/{model_filename}.pt'):
        
            # Train model
            kwargs = {'sg': sg,
                    'timestep': timestep}

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
                                loss=loss,
                                device=device,
                                epochs=epochs,
                                **kwargs)
                    end = time.time()
                    print(f'Elapsed time : {end-start}s')

                    # Save model
                    torch.save(model, f'{saved_model_path}/{model_filename}.pt')

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
                                loss=loss,
                                device=device,
                                epochs=epochs,
                                **kwargs)
                    end = time.time()
                    print(f'Elapsed time : {end-start}s')

                    # TODO : Save list of model for GCN lc
                    #torch.save(model, f'{saved_model_path}/{model_filename}.pt')

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
                                    loss=loss,
                                    device=device,
                                    epochs=epochs,
                                    **kwargs)
                        end = time.time()
                        trained_models.append(model_full)
                        print(f'Elapsed time : {end-start}s')

                        # TODO : Save list of model for GCN lc
                        #torch.save(model, f'{saved_model_path}/{model_filename}.pt')

        # Use pre-trained model
        else:
            print('\nModel have already been trained for selection of parameters in input.')
            if model_name != 'GCN_lc':
                print(f'Pre-trained model used for evaluation : {saved_model_path}/{model_filename}.pt')
                models = [torch.load(f'{saved_model_path}/{model_filename}.pt')]
            else:
                # TODO : load list of models when linear combination of GCNs are trained
                pass
            

        # ------ Evaluation ------

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

            if test_agg == 'True':
                # Full evaluation dataset
                print(f'VAL POS G : {sg.val_pos_g}')
                print(f'VAL NEG G : {sg.val_neg_g}')
                history_score, val_pos_score, val_neg_score = trained_model.test(pred, 
                                                                    sg.val_pos_g, 
                                                                    sg.val_neg_g, 
                                                                    metric=metric, 
                                                                    timestep=timestep,
                                                                    feat_struct=feat_struct, 
                                                                    step_prediction=step_prediction,
                                                                    k_indexes=k_indexes,
                                                                    sg=sg,
                                                                    model_name=model_name,
                                                                    shuffle_test=shuffle_test,
                                                                    return_all=True)                                                

                if len(models) > 1:
                    label = f'{triplets[idx]}'
                else:
                    label = f'{model_name}'

                df_tmp = pd.DataFrame([[label, history_score[f'test_{metric}'], 0, sg.val_pos_g.number_of_edges(), sg.val_neg_g.number_of_edges(), test_agg, \
                                        dup_edges, predictor, loss_func]], 
                                        columns=['model', 'score', 'timestep', 'number_of_edges_pos', 'number_of_edges_neg', 'test_agg', \
                                                'duplicate_edges', 'predictor', 'loss_func'])
                df_tot = pd.concat([df_tot, df_tmp])

            elif test_agg == 'False':
                # Step evaluation dataset
                for val_pos_g, val_neg_g, t in zip(val_pos_g_list, val_neg_g_list, sg.trange_val):
                    if val_pos_g.number_of_edges() > 0 and val_neg_g.number_of_edges() > 0 and t != 0:
                        history_score, val_pos_score, val_neg_score = trained_model.test(pred, 
                                                                            val_pos_g, 
                                                                            val_neg_g, 
                                                                            metric=metric,
                                                                            timestep=timestep,
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

                        df_tmp = pd.DataFrame([[label, history_score[f'test_{metric}'], t, val_pos_g.number_of_edges(), val_neg_g.number_of_edges(), test_agg, \
                                                dup_edges, predictor, loss_func]], 
                                                columns=['model', 'score', 'timestep', 'number_of_edges_pos', 'number_of_edges_neg', 'test_agg', \
                                                'duplicate_edges', 'predictor', 'loss_func'])
                        df_tot = pd.concat([df_tot, df_tmp])

                        #plot_history_loss(hist_train_loss, ax=ax[0], label=label)
                        #ax[0].set_xlabel('epochs')       
                        #plot_result(history_score, ax=ax[1], title='Eval set - unseen nodes', label=label, metric=metric)

        print('Done !')

    # ------ Save results ------

    res_path = f'{result_path}/{data}/{feat_struct}'
    if feat_struct == 'temporal_edges':
        res_filename = f'{data}_{model_name}_{feat_struct}_{metric}_{test_agg}_{dup_edges}_{predictor}_{loss_func}_{step_prediction}_{shuffle_test}'
    elif feat_struct == 'baseline':
        res_filename = f'{data}_{model_name}_{feat_struct}_{metric}_{shuffle_test}'
    else:
        res_filename = f'{data}_{model_name}_{feat_struct}_{metric}_{test_agg}_{dup_edges}_{predictor}_{loss_func}_{shuffle_test}'

    df_tot.to_pickle(f'{res_path}/{res_filename}.pkl', protocol=3)
    print(f'Results saved in {res_path}/{res_filename}.pkl')
    print(df_tot.shape)


if __name__=='__main__':

    LOG_PATH = f'{os.getcwd()}/logs'
    #RESULT_PATH = f'{os.getcwd()}/results'
    RESULT_PATH = '/home/infres/sdelarue/node-embedding/GNN/results'
    MODEL_PATH = '/home/infres/sdelarue/node-embedding/GNN/models'

    parser = argparse.ArgumentParser('Preprocessing data')
    parser.add_argument('--data', type=str, help='Dataset name : \{SF2H, HighSchool, ia-contact, ia-contacts_hypertext2009, fb-forum, ia-enron-employees\}', default='SF2H')
    parser.add_argument('--cache', type=str, help='Path for splitted graphs already cached', default=None)
    parser.add_argument('--feat_struct', type=str, help='Data structure : \{agg, agg_simp, time_tensor, temporal_edges, DTFT, baseline\}', default='time_tensor')
    parser.add_argument('--step_prediction', type=str, help="If data structure is 'temporal_edges', either 'single' or 'multi' step predictions can be used.", default='single')
    parser.add_argument('--normalized', type=bool, help='If true, normalized adjacency is used', default=True)
    parser.add_argument('--model', type=str, help='GCN model : \{GraphConv, GraphSage, GCNTime, baseline_avg\}', default='GraphConv')
    parser.add_argument('--batch_size', type=int, help='If batch_size > 0, stream graph is splitted into batches.', default=0)
    parser.add_argument('--timestep', type=int, help='Finest granularity in temporal data.', default=20)
    parser.add_argument('--emb_size', type=int, help='Embedding size', default=20)
    parser.add_argument('--epochs', type=int, help='Number of epochs in training', default=1)
    parser.add_argument('--lr', type=float, help='Learning rate in for training', default=0.001)
    parser.add_argument('--metric', type=str, help='Evaluation metric : \{auc, kendall, wkendall, spearmanr, \{kendall, wkendall, spearmanr\}@\{5, 10, 25, 50, 100\}\}', default='auc')
    parser.add_argument('--duplicate_edges', type=str, help='If true, allows duplicate edges in training graphs', default='True')
    parser.add_argument('--test_agg', type=str, help='If true, predictions are performed on a static graph test.', default='True')
    parser.add_argument('--predictor', type=str, help='\{dotProduct, cosine\}', default='dotProduct')
    parser.add_argument('--loss_func', type=str, help='\{BCEWithLogits, graphSage, marginRanking, torchMarginRanking, pairwise\}', default='BCEWithLogits')
    parser.add_argument('--shuffle_test', type=str, help='If True, shuffle test set links order.', default='False')
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
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    DUP_EDGES = args.duplicate_edges
    TEST_AGG = args.test_agg
    PREDICTOR = args.predictor
    LOSS_FUNC = args.loss_func
    SHUFFLE_TEST = args.shuffle_test

    print(f'Device : {DEVICE}')
    
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
        MODEL_PATH,
        DUP_EDGES,
        TEST_AGG,
        PREDICTOR,
        LOSS_FUNC,
        SHUFFLE_TEST)


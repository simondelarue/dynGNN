# -*- coding: utf-8 -*-

import pandas as pd
import torch
import dgl
import os

def load_data(dataset):
    print(os.getcwd())
    PATH = f'{os.getcwd()}/data'

    if dataset == 'SF2H':
        df_data = pd.read_csv(f'{PATH}/tij_SFHH.dat_', header=None, names=['t', 'i', 'j'], delimiter=' ')

    return df_data

def save_data(data_df, dir, name):
    data_df.to_pickle(f'{dir}/{name}.pkl')

def preprocess(df, dataset):

    print('Preprocessing data ...')
    if dataset == 'SF2H':

        # Reindex node labels
        df_preproc = df.copy()
        unique_nodes = set(df_preproc['i'].values) | set(df_preproc['j'].values)

        mapping = {}
        for idx, node in enumerate(unique_nodes):
            mapping[node] = idx

        df_preproc['src'] = df_preproc['i'].apply(lambda x: mapping[x])
        df_preproc['dest'] = df_preproc['j'].apply(lambda x: mapping[x])

        return df_preproc

def temporal_graph(dataset):
    PREPROC_PATH = f'{os.getcwd()}/preprocessed_data'

    # Load and save data
    data_df = load_data(dataset)
    preproc_data_df = preprocess(data_df, dataset)
    save_data(preproc_data_df, PREPROC_PATH, dataset)

    # Create temporal graph
    print('Creating temporal graph ...')
    g = dgl.graph((preproc_data_df['src'], preproc_data_df['dest']))
    g.edata['timestamp'] = torch.from_numpy(preproc_data_df['t'].to_numpy())

    print('Done !')
    return g

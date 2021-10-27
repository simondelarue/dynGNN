# -*- coding: utf-8 -*-
import pandas as pd
from os import listdir
import matplotlib.pyplot as plt
import numpy as np

def plot_val_score(x, y, ax, metric, label):
    ax.plot(x, y, label=label)
    ax.set_xlabel('timestep')
    ax.set_ylabel(f'{metric}')
    ax.legend()

if __name__=='__main__':
    global_path = '/home/infres/sdelarue/node-embedding/GNN/results'

    datasets = ['SF2H', 'HighSchool', 'ia-contact', 'ia-contacts_hypertext2009', 'ia-enron-employees']
    methods = ['agg_simp', 'agg', 'temporal_edges']
    #step_predictions = ['single', 'multi']

    df = pd.DataFrame()
    for dataset in datasets:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

        for method in methods:
            path = f'{dataset}/{method}'
            files = [f for f in listdir(f'{global_path}/{path}') if (f.endswith('.pkl') and not f.startswith(f'{dataset}_GCN_lc'))]

            for f in files:
                df_tmp = pd.read_pickle(f'{global_path}/{path}/{f}')
                df_tmp['method'] = method
                df_tmp['dataset'] = dataset
                df = pd.concat([df, df_tmp])

    df.to_csv(f'{global_path}/results.csv', index=False)

    # Average results over time
    df_avg = df.copy()
    cols = ['dataset', 'model', 'method', 'score', 'timestep', 'test_agg', 'duplicate_edges', 'predictor', 'loss_func']
    agg_cols = ['dataset', 'model', 'method', 'test_agg', 'duplicate_edges', 'predictor', 'loss_func']

    # Whole test - no duplicate edges
    df_avg_1 = df_avg[(df_avg['test_agg']=='True') & (df_avg['duplicate_edges']=='False')]
    df_avg_1 = df_avg_1[cols].groupby(agg_cols)['score'].mean().reset_index()
    df_avg_1 = (pd.pivot_table(df_avg_1, values=['score'], index='dataset', columns=['method', 'model'])*100).round(2)
    df_avg_1.to_csv(f'{global_path}/results_test_agg_no_dup_edges.csv')

    # Whole test - duplicate edges
    df_avg_2 = df_avg[(df_avg['test_agg']=='True') & (df_avg['duplicate_edges']=='True')]
    df_avg_2 = df_avg_2[cols].groupby(agg_cols)['score'].mean().reset_index()
    df_avg_2 = (pd.pivot_table(df_avg_2, values=['score'], index='dataset', columns=['method', 'model'])*100).round(2)
    df_avg_2.to_csv(f'{global_path}/results_test_agg_dup_edges.csv')

    # Snapshots test - no duplicate edges
    df_avg_3 = df_avg[(df_avg['test_agg']=='False') & (df_avg['duplicate_edges']=='False')]
    df_avg_3 = df_avg_3[cols].groupby(agg_cols)['score'].mean().reset_index()
    df_avg_3 = (pd.pivot_table(df_avg_3, values=['score'], index='dataset', columns=['method', 'model'])*100).round(2)
    df_avg_3.to_csv(f'{global_path}/results_no_test_agg_no_dup_edges.csv')

    # Snapshots test - duplicate edges
    df_avg_4 = df_avg[(df_avg['test_agg']=='False') & (df_avg['duplicate_edges']=='True')]
    print(df_avg_4['dataset'].unique())
    tmp = (df_avg_4[(df_avg_4['model']=='GraphConv') & \
                   (df_avg_4['dataset']=='ia-contacts_hypertext2009') & \
                   (df_avg_4['method']=='agg_simp')])
    print(tmp.head())
    df_avg_4 = df_avg_4[cols].groupby(agg_cols)['score'].mean().reset_index()
    df_avg_4 = (pd.pivot_table(df_avg_4, values=['score'], index='dataset', columns=['method', 'model'])*100).round(2)
    df_avg_4.to_csv(f'{global_path}/results_no_test_agg_dup_edges.csv')    
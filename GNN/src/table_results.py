# -*- coding: utf-8 -*-
import argparse
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
    parser = argparse.ArgumentParser('Preprocessing data')
    parser.add_argument('--model', type=str, help='Model : \{GCN_lc\}', default=None)
    args = parser.parse_args()

    global_path = '/home/infres/sdelarue/node-embedding/GNN/results'

    #datasets = ['SF2H', 'HighSchool', 'ia-contact', 'ia-contacts_hypertext2009', 'ia-enron-employees']
    datasets = ['SF2H', 'HighSchool', 'ia-contacts_hypertext2009']
    methods = ['agg_simp', 'agg', 'temporal_edges', 'time_tensor', 'DTFT']
    #step_predictions = ['single', 'multi']

    df = pd.DataFrame()
    for dataset in datasets:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

        for method in methods:
            path = f'{dataset}/{method}'

            if args.model == 'GCN_lc':
                files = [f for f in listdir(f'{global_path}/{path}') if (f.endswith('.pkl') and f.startswith(f'{dataset}_GCN_lc'))]
            else:
                files = [f for f in listdir(f'{global_path}/{path}') if (f.endswith('.pkl') and not f.startswith(f'{dataset}_GCN_lc'))]

            for f in files:
                df_tmp = pd.read_pickle(f'{global_path}/{path}/{f}')
                df_tmp['method'] = method
                df_tmp['dataset'] = dataset
                for metric in ['auc', 'kendall', 'wkendall']:
                    if metric in f:
                        df_tmp['metric'] = metric
                        if metric == 'auc':
                            df_tmp['score'] = (df_tmp['score']*100).round(2)
                        elif metric in ['kendall', 'wkendall']:
                            df_tmp['score'] = (df_tmp['score']).round(4)
                
                df = pd.concat([df, df_tmp])

    filename = 'results'
    if args.model == 'GCN_lc':
        filename = 'results_GCN_lc'

    # Save results in .csv
    df.to_csv(f'{global_path}/{filename}.csv', index=False)

    # Average results over time
    df_avg = df.copy()
    cols = ['dataset', 'model', 'method', 'metric', 'score', 'timestep', 'test_agg', 'duplicate_edges', 'predictor', 'loss_func']
    agg_cols = ['dataset', 'model', 'method', 'metric', 'test_agg', 'duplicate_edges', 'predictor', 'loss_func']

    # Whole test - no duplicate edges
    filename1 = 'results_test_agg_no_dup_edges'
    if args.model == 'GCN_lc':
        filename1 = 'results_test_agg_no_dup_edges_GCN_lc'
    df_avg_1 = df_avg[(df_avg['test_agg']=='True') & (df_avg['duplicate_edges']=='False') & (df_avg['metric']=='kendall')]
    df_avg_1 = df_avg_1[cols].groupby(agg_cols)['score'].mean().reset_index()
    df_avg_1 = pd.pivot_table(df_avg_1, values=['score'], index='dataset', columns=['method', 'model', 'metric'])
    df_avg_1.to_csv(f'{global_path}/{filename1}_filt.csv')
    print(df_avg_1)

    # Whole test - duplicate edges
    filename2 = 'results_test_agg_dup_edges'
    if args.model == 'GCN_lc':
        filename2 = 'results_test_agg_dup_edges_GCN_lc'
    df_avg_2 = df_avg[(df_avg['test_agg']=='True') & (df_avg['duplicate_edges']=='True') & (df_avg['metric']=='kendall')]
    df_avg_2 = df_avg_2[cols].groupby(agg_cols)['score'].mean().reset_index()
    df_avg_2 = pd.pivot_table(df_avg_2, values=['score'], index='dataset', columns=['method', 'model', 'metric'])
    df_avg_2.to_csv(f'{global_path}/{filename2}_filt.csv')
    print(df_avg_2)

    # Snapshots test - no duplicate edges
    '''filename3 = 'results_no_test_agg_no_dup_edges'
    if args.model == 'GCN_lc':
        filename3 = 'results_no_test_agg_no_dup_edges_GCN_lc'
    df_avg_3 = df_avg[(df_avg['test_agg']=='False') & (df_avg['duplicate_edges']=='False')]
    df_avg_3 = df_avg_3[cols].groupby(agg_cols)['score'].mean().reset_index()
    df_avg_3 = pd.pivot_table(df_avg_3, values=['score'], index='dataset', columns=['method', 'model', 'metric'])
    df_avg_3.to_csv(f'{global_path}/{filename3}.csv')

    # Snapshots test - duplicate edges
    filename4 = 'results_no_test_agg_dup_edges'
    if args.model == 'GCN_lc':
        filename4 = 'results_no_test_agg_dup_edges_GCN_lc'
    df_avg_4 = df_avg[(df_avg['test_agg']=='False') & (df_avg['duplicate_edges']=='True')]
    print(df_avg_4['dataset'].unique())
    tmp = (df_avg_4[(df_avg_4['model']=='GCNTime') & \
                   (df_avg_4['dataset']=='ia-contacts_hypertext2009') & \
                   (df_avg_4['method']=='DTFT')])
    print(tmp.head())
    df_avg_4 = df_avg_4[cols].groupby(agg_cols)['score'].mean().reset_index()
    df_avg_4 = pd.pivot_table(df_avg_4, values=['score'], index='dataset', columns=['method', 'model', 'metric'])
    df_avg_4.to_csv(f'{global_path}/{filename4}.csv')   ''' 
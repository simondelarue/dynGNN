# -*- coding: utf-8 -*-
import pandas as pd
from os import listdir
import matplotlib.pyplot as plt


def plot_val_score(x, y, ax, metric, label):
    ax.plot(x, y, label=label)
    ax.set_xlabel('timestep')
    ax.set_ylabel(f'{metric}')
    ax.legend()

if __name__=='__main__':
    global_path = '/home/infres/sdelarue/node-embedding/GNN/results'

    datasets = ['SF2H', 'HighSchool', 'ia-contact', 'ia-contacts_hypertext2009', 'ia-enron-employees']
    methods = ['agg_simp', 'temporal_edges']
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
                df = pd.concat([df, df_tmp])

    df.to_csv(f'{global_path}/results.csv', index=False)

    # Average results over time
    df_avg = df.copy()
    cols = ['model', 'method', 'score', 'timestep', 'test_agg', 'duplicate_edges', 'predictor', 'loss_func']
    agg_cols = ['model', 'method', 'test_agg', 'duplicate_edges', 'predictor', 'loss_func']

    # Whole test - no duplicate edges
    df_avg_1 = df_avg[(df_avg['test_agg']=='True') & (df_avg['duplicate_edges']=='False')]
    df_avg_1 = df_avg_1[cols].groupby(agg_cols)['score'].mean()
    df_avg_1.to_csv(f'{global_path}/results_test_agg_no_dup_edges.csv')

    # Whole test - duplicate edges
    df_avg_2 = df_avg[(df_avg['test_agg']=='True') & (df_avg['duplicate_edges']=='True')]
    df_avg_2 = df_avg_2[cols].groupby(agg_cols)['score'].mean()
    df_avg_2.to_csv(f'{global_path}/results_test_agg_dup_edges.csv')

    # Snapshots test - no duplicate edges
    df_avg_3 = df_avg[(df_avg['test_agg']=='False') & (df_avg['duplicate_edges']=='False')]
    df_avg_3 = df_avg_3[cols].groupby(agg_cols)['score'].mean()
    df_avg_3.to_csv(f'{global_path}/results_no_test_agg_no_dup_edges.csv')

    # Snapshots test - duplicate edges
    df_avg_4 = df_avg[(df_avg['test_agg']=='False') & (df_avg['duplicate_edges']=='True')]
    df_avg_4 = df_avg_4[cols].groupby(agg_cols)['score'].mean()
    df_avg_4.to_csv(f'{global_path}/results_no_test_agg_dup_edges.csv')
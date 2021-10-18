# -*- coding: utf-8 -*-
import pandas as pd
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from utils import save_figures

def plot_val_score(x, y, ax, metric, label):
    ax.plot(x, y, label=label)
    ax.set_xlabel('timestep')
    ax.set_ylabel(f'{metric}')
    ax.legend()

if __name__=='__main__':
    global_path = '/home/infres/sdelarue/node-embedding/GNN/results'

    #datasets = ['SF2H', 'HighSchool', 'ia-contact']
    datasets = ['ia-enron-employees']
    methods = ['agg', 'temporal_edges']
    #methods = ['agg', 'temporal_edges', 'time_tensor']
    step_predictions = ['single', 'multi']

    for dataset in datasets:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

        for method in methods:
            path = f'{dataset}/{method}'
            files = [f for f in listdir(f'{global_path}/{path}') if (f.endswith('.pkl') and not f.startswith(f'{dataset}_GCN_GCN_lc'))]

            for f in files:
                df_tmp = pd.read_pickle(f'{global_path}/{path}/{f}')

                # Plot results
                avg_score = np.mean(df_tmp['score'])
                if method=='temporal_edges':
                    step_pred = f[-10:-4]
                    label = f"{df_tmp['model'].unique()[0]} - {method} - {step_pred} - Avg AUC={avg_score:.3f}"
                else:
                    label = f"{df_tmp['model'].unique()[0]} - {method} - Avg AUC={avg_score:.3f}"

                plot_val_score(#np.array(df_tmp['timestep'])
                                range(len(np.array(df_tmp['score']))),
                                np.array(df_tmp['score']),
                                ax=ax,
                                metric='ROC AUC',
                                label=label)

        # Save results
        filename = f"{dataset}_all_timesteps"
        save_figures(fig, f'{global_path}', filename)


    # GCN Linear combination
    '''for dataset in datasets:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

        path = f'{dataset}/agg'
        files = [f for f in listdir(f'{global_path}/{path}') if (f.startswith(f'{dataset}_GCN_GCN_lc') and f.endswith('.pkl'))]

        for f in files:
            df_tmp = pd.read_pickle(f'{global_path}/{path}/{f}')

            for model in df_tmp['model'].unique():
                df_tmp_model = df_tmp[df_tmp['model']==model]
                # Plot results
                avg_score = np.mean(df_tmp_model['score'])
                label = f"{model} - agg - Avg AUC={avg_score:.2f}"

                plot_val_score(np.array(df_tmp_model['timestep']),
                                np.array(df_tmp_model['score']),
                                ax=ax,
                                metric='ROC AUC',
                                label=label)

        # Save results
        filename = f"{dataset}_GCN_lc_all_timesteps"
        save_figures(fig, f'{global_path}', filename)'''

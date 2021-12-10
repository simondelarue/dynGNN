# -*- coding: utf-8 -*-
import pandas as pd
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from utils import save_figures

def plot_val_score(x, y, ax, metric, marker, label):
    ax.plot(x, y, label=label, marker=marker, alpha=0.8)
    ax.set_xlabel('timestep')
    ax.set_ylabel(f'{metric}')
    ax.legend()

if __name__=='__main__':
    global_path = '/home/infres/sdelarue/node-embedding/GNN/results'

    #datasets = ['SF2H', 'HighSchool', 'ia-contacts_hypertext2009']
    datasets = ['SF2H']
    methods = ['agg_simp', 'agg', 'temporal_edges', 'time_tensor', 'DTFT']
    order = {'@5': 0, '@10': 1, '@25': 2, '@50': 3, '@100': 4}
    titles = {'agg_simp': 'Agg', 'agg': 'wAgg', 'temporal_edges': 'TDAG', 'time_tensor': '3d-tensor', 'DTFT': 'DTFT'}
    items = {'GraphConv': ['black', '*'], 'GraphSage': ['darkblue', '.'], 'GCNTime': ['green', '+']}
    metric = 'Spearmanr'
    #step_predictions = ['single', 'multi']

    for i, dataset in enumerate(datasets):
        fig, ax = plt.subplots(1, len(methods), figsize=(20, 4))
        
        for j, method in enumerate(methods):
            x, y = [], []
            path = f'{dataset}/{method}'

            files = [f for f in listdir(f'{global_path}/{path}') if (f.endswith('.pkl') and (metric.lower() in f) and ('@' in f) and ('kendall' not in f) and ('pairwise' in f) and ('cosine' in f) and not f.startswith(f'{dataset}_GCN_lc'))]
            df_tot = pd.DataFrame()
            for f in files:
                df_tmp = pd.read_pickle(f'{global_path}/{path}/{f}')
                df_tmp['method'] = method
                df_tmp['dataset'] = dataset
                df_tmp['metric'] = '@' + f.split('@')[1].split('_')[0]
                df_tmp['order'] = df_tmp['metric'].apply(lambda x: order.get(x))
                pred = df_tmp['predictor'].unique()[0]
                loss = df_tmp['loss_func'].unique()[0]
                df_tot = pd.concat([df_tot, df_tmp])

            df_tot = df_tot.sort_values(['model', 'order'])

            models = np.array(df_tot['model'].unique())
            for m, model in enumerate(models):
                if dataset == 'ia-contacts_hypertext2009' and len(np.array(df_tot[df_tot['model']==model]['metric']))>3:
                    ax[j].plot(np.array(df_tot[df_tot['model']==model]['metric'])[2:], 
                            np.array(df_tot[df_tot['model']==model]['score'])[2:], 
                            label=model, marker=items.get(model)[1], color=items.get(model)[0])
                else:
                    ax[j].plot(np.array(df_tot[df_tot['model']==model]['metric']), 
                                np.array(df_tot[df_tot['model']==model]['score']), 
                                label=model, marker=items.get(model)[1], color=items.get(model)[0])
                ax[j].legend(loc='upper right')
                ax[j].set_xlabel(f'{metric}@')
                if metric == 'Kendall':
                    var = 'tau'
                elif metric == 'Spearmanr':
                    var = 'rho'
                ax[j].set_ylabel(f'{metric} {var}')
                ax[j].set_ylim(-0.1, 0.3)
                ax[j].set_title(titles.get(method), weight='bold')
                ax[j].spines['top'].set_visible(False)
                ax[j].spines['right'].set_visible(False)
                ax[j].spines['bottom'].set_visible(True)
                ax[j].spines['left'].set_visible(True)

        # Save results
        filename = f"{dataset}_True_True_{metric}@_{pred}_{loss}"
        fig.savefig(f'{global_path}/{dataset}/{filename}.eps', bbox_inches='tight', transparent=False, pad_inches=0)    
        save_figures(fig, f'{global_path}/{dataset}', filename)


    '''for dataset in datasets:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        markers = ['*', '.', '+']
        for method, marker in zip(methods, markers):
            path = f'{dataset}/{method}'
            files = [f for f in listdir(f'{global_path}/{path}') if (f.endswith('.pkl') and not f.startswith(f'{dataset}_GCN_lc'))]

            for f in files:
                df_tmp = pd.read_pickle(f'{global_path}/{path}/{f}')
                model = df_tmp['model'].unique()[0]
                test_agg = df_tmp['test_agg'].unique()[0]
                dup_edge = df_tmp['duplicate_edges'].unique()[0]
                pred = df_tmp['predictor'].unique()[0]
                loss = df_tmp['loss_func'].unique()[0]

                # Plot results
                avg_score = np.mean(df_tmp['score'])
                if method=='temporal_edges':
                    step_pred = f[-10:-4]
                    label = f"{df_tmp['model'].unique()[0]} - {method} - {step_pred} - Avg AUC={100*(avg_score):.2f}"
                else:
                    if method == 'agg_simp':
                        method_name = 'wAgg'
                    elif method == 'agg':
                        method_name = 'Agg'
                    elif method == 'time_tensor':
                        method_name = '3d-tensor'
                    label = f"{method_name} - {model} - Avg AUC={100*(avg_score):.2f}"

                if model in ['GraphConv', 'GCNTime'] and dup_edge == 'True' and test_agg == 'False':
                    plot_val_score(#np.array(df_tmp['timestep'])
                                    range(len(np.array(df_tmp['score']))),
                                    np.array(df_tmp['score']),
                                    ax=ax,
                                    metric='ROC AUC',
                                    marker=marker,
                                    label=label)

        # Save results
        filename = f"{dataset}_False_True_{pred}_{loss}"
        save_figures(fig, f'{global_path}/{dataset}', filename)'''


    # GCN Linear combination
    '''for dataset in datasets:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

        path = f'{dataset}/agg'
        files = [f for f in listdir(f'{global_path}/{path}') if (f.startswith(f'{dataset}_GCN_lc_agg_auc_False') and f.endswith('.pkl'))]

        for f in files:
            df_tmp = pd.read_pickle(f'{global_path}/{path}/{f}')
            dup_edge = df_tmp['duplicate_edges'].unique()[0]
            pred = df_tmp['predictor'].unique()[0]
            loss = df_tmp['loss_func'].unique()[0]
            
            df_top_3 = df_tmp[['model', 'score']].groupby('model')['score'].mean().reset_index()
            top_3_avg_scores = sorted(df_top_3['score'], reverse=True)[:3]

            for model in df_tmp['model'].unique():
                df_tmp_model = df_tmp[df_tmp['model']==model]
                # Plot results
                avg_score = np.mean(df_tmp_model['score'])
                label = f"{model} - agg - Avg AUC={100*(avg_score):.2f}"

                if avg_score >= np.min(top_3_avg_scores):
                    plot_val_score(np.array(df_tmp_model['timestep']),
                                    np.array(df_tmp_model['score']),
                                    ax=ax,
                                    metric='ROC AUC',
                                    label=label)

            # Save results
            filename = f"{dataset}_GCN_lc_False_{dup_edge}_{pred}_{loss}"
            save_figures(fig, f'{global_path}/{dataset}/agg', filename)'''

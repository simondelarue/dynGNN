# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


if __name__=='__main__':
    global_path = '/home/infres/sdelarue/node-embedding/GNN/results'

    datasets = ['SF2H', 'HighSchool', 'ia-contact', 'ia-contacts_hypertext2009', 'ia-enron-employees']
    
    df = pd.read_csv(f'{global_path}/results.csv')

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    markers = ['x', '.', '1', '_', '+', '*']
    for idx, dataset in enumerate(datasets):
        df_tmp = df[(df['dataset']==dataset) & (df['test_agg']==False)][['timestep', 'number_of_edges_pos', 'number_of_edges_neg']].drop_duplicates()
        avg_nb_edges = np.mean(df_tmp['number_of_edges_pos'])
        plt.plot(range(100), df_tmp['number_of_edges_pos'].to_numpy()[:100], label=f'{dataset} (avg $|E|$={avg_nb_edges:.1f})', marker=markers[idx], alpha=0.6)

    plt.legend(loc='upper right')
    plt.xlabel('$T$')
    plt.ylabel('$|E|$')
    fig.savefig(f'{global_path}/img/nb_edges.png')

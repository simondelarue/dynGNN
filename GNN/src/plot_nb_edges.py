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
    
    df = pd.read_csv(f'{global_path}/results.csv')

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))

    for dataset in datasets:
        df_tmp = df[(df['dataset']==dataset)][['timestep', 'number_of_edges_pos', 'number_of_edges_neg']].drop_duplicates()
        if dataset in ['SF2H', 'HighSchool', 'ia-enron-employees']:
            print(df_tmp.sort_values('number_of_edges_pos', ascending=False).head())
        avg_nb_edges = np.mean(df_tmp['number_of_edges_pos'])
        plt.plot(range(1, 100), df_tmp['number_of_edges_pos'].to_numpy()[1:100], label=f'{dataset} (mean={avg_nb_edges:.1f})')

    plt.legend(loc='upper right')
    plt.xlabel('$T$')
    plt.ylabel('$|E|$')
    fig.savefig(f'{global_path}/img/nb_edges.png')

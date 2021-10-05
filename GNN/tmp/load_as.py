# -*- coding: utf-8 -*-

import pandas as pd
from os import listdir

def load_data(path):
    # Load data
    filepaths = [f for f in listdir(path) if f.endswith('.txt')]
    files = sorted(filepaths)[-100:]

    df_tot = pd.DataFrame()

    index_f = {}
    for t, f in enumerate(files):
        index_f[f] = index_f.get(f, t)

    for f in files:
        df_tmp = pd.read_csv(f'{path}/{f}', 
                            delimiter='\t',
                            header=3,
                            names=['i', 'j'])
        df_tmp['t'] = index_f[f]
        df_tot = pd.concat([df_tot, df_tmp])
    
    return df_tot

def save_data(df, path, filename)
    # Save results
    df.to_pickle(f'{path}/{filename}.pkl', protocol=3)

if __name__='main':
    path = '/Users/simondelarue/Downloads/as-733'
    df = load_data(path)
    save_data(df, path, as_100)
# -*- coding: utf-8 -*-
import pandas as pd
from os import listdir

if __name__=='__main__':

    global_path = '/home/infres/sdelarue/node-embedding/GNN/results'

    datasets = ['SF2H', 'HighSchool', 'ia-contacts_hypertext2009']
    methods = ['baseline']
    metric = 'kendall'
    columns = ['@' + val for val in ['5', '10', '25', '50', '100']]

    for dataset in datasets:
         
        df = pd.DataFrame()

        for method in methods:

            path = f'{dataset}/{method}'

            files = []
            for f in listdir(f'{global_path}/{path}'):
                # select shuffled test links order files
                if f.startswith(f'{dataset}_{method}') and f.endswith('True.pkl'):
                    files.append(f)

            for f in files:
                df_tmp = pd.read_pickle(f'{global_path}/{path}/{f}')
                df_tmp['method'] = method
                df_tmp['dataset'] = dataset
                df_tmp['metric'] = '@' + f.split('@')[1].split('_')[0]
                df_tmp['score'] = (df_tmp['score']).round(4)

                df = pd.concat([df, df_tmp])

        df_piv = pd.pivot_table(df, values=['score'], index=['method', 'model'], columns=['metric'])

        filename = f'results_{dataset}_{metric}'

        # Reindex rows and columns according to desired display
        new_cols = df_piv.columns.reindex(columns, level=1)
        df_piv = df_piv.reindex(columns=new_cols[0])
        
        new_rows = df_piv.index.reindex(methods, level=0)
        df_piv = df_piv.reindex(new_rows[0])

        # Save results in .csv
        print(df_piv)
        df_piv.to_csv(f'{global_path}/2_ranking/{filename}_{method}_shuffled.csv', index=True)

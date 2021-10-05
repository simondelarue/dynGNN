import os
import pandas as pd

class DataLoader():
    ''' Loads and preprocesses data according to the desired source.

        Parameters
        -----------
            dataset: str
                Name of the dataset to load.
            in_path: str (default = 'data')
                Local path where to find data.
            out_path: str (default = 'preprocessed_data')
                Local path where preprocessed data should be saved. '''

    def __init__(self, dataset: str, in_path: str = 'data', out_path: str = 'preprocessed_data'):
        self.name = dataset
        self.IN_PATH = f'{os.getcwd()}/{in_path}'
        self.OUT_PATH = f'{os.getcwd()}/{out_path}'

        # Preprocess data
        self.data_df = self.__preprocess(self.__load())
        # Save preprocessed data
        self.__save(self.name)


    def __load(self) -> pd.DataFrame:
        ''' Load data from source file. 
            Output
            -------
                DataFrame '''

        if self.name == 'SF2H':
            return pd.read_csv(f'{self.IN_PATH}/tij_SFHH.dat_', header=None, names=['t', 'i', 'j'], delimiter=' ')
        elif self.name == 'HighSchool':
            return pd.read_csv(f'{self.IN_PATH}/High-School_data_2013.csv', header=None, names=['t', 'i', 'j', 'Ci', 'Cj'], delimiter=' ')
        elif self.name == 'AS':
            print(f'{self.IN_PATH}/as-733/as_100.pkl')
            return pd.read_pickle(f'{self.IN_PATH}/as-733/as_100.pkl')


    def __save(self, out_name: str):
        ''' Save data in `pickle` format.
            Parameters
            -----------
                out_name : str
                    Label of dataset '''

        self.data_df.to_pickle(f'{self.OUT_PATH}/{out_name}.pkl', protocol=3)


    def __preprocess(self, data_df: pd.DataFrame) -> pd.DataFrame:
        ''' Preprocessing consists of reindexing nodes labels from 0 to |V|.
        
            Parameters
            -----------
                data_df: DataFrame
                    Temporal data as a dataframe containing at least triplets (t, i, j).
                    
            Output
            -------
                DataFrame with reindexed node labels. '''

        print('Preprocessing data ...')

        if self.name in ['SF2H', 'HighSchool', 'AS']:

            # Reindex node labels 
            df_preproc = data_df.copy()
            unique_nodes = set(df_preproc['i'].values) | set(df_preproc['j'].values)

            mapping = {}
            for idx, node in enumerate(unique_nodes):
                mapping[node] = idx

            df_preproc['src'] = df_preproc['i'].apply(lambda x: mapping[x])
            df_preproc['dest'] = df_preproc['j'].apply(lambda x: mapping[x])
            
            return df_preproc



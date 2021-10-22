#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Oct 20, 2021
@author: Simon Delarue <sdelarue@enst.fr>
'''

import pandas as pd
from os.path import expanduser
from typing import Optional, Union
from pathlib import Path

import numpy as np
from scipy import sparse

from stream.utils import Bunch
from stream.data.parse import convert_edge_list


def load_data(dataset: Optional[str] = None, data_home: Optional[Union[str, Path]] = None) -> Bunch:
    
    graph = Bunch()
    data_home = Path(expanduser(data_home)) 
    data_path = data_home / dataset

    if dataset == 'ml-latest-small':
        df = pd.read_csv(data_path / 'ratings.csv')

        edge_list = list(df.itertuples(index=False))
        graph = convert_edge_list(edge_list, bipartite=True)

    return graph
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Oct 20, 2021
@author: Simon Delarue <sdelarue@enst.fr>

This code is heavily inspired from scikit-network package.
'''

import numpy as np
from scipy import sparse

from typing import Optional, List, Tuple, Union

from stream.utils import Bunch

def convert_edge_list(edge_list: Union[np.ndarray, List[Tuple], List[List]], reindex: bool = False, bipartite: bool = False) -> Bunch:
    ''' Turn an edge list into a :class:`Bunch`. '''
    
    if isinstance(edge_list, list):
        edge_list = np.array(edge_list)
    else:
        raise TypeError('Edge lists must be given as NumPy arrays or lists of lists or lists of tuples.')

    row, col, data = edge_list[:, 0], edge_list[:, 1], np.array([])

    return from_edge_list(row=row, col=col, data=data, reindex=reindex, bipartite=bipartite)

def from_edge_list(row: np.ndarray, col: np.ndarray, data: np.ndarray, reindex: bool = False, bipartite: bool = False) -> Bunch:
    ''' Turn triplet of Numpy arrays (row, col, data) as a :class:`Bunch`. '''

    graph = Bunch()

    if bipartite:
        names_row, row = np.unique(row, return_inverse=True)
        names_col, col = np.unique(col, return_inverse=True)
        n_row = len(names_row)
        n_col = len(names_col)
        data = np.ones(len(row))
        biadjacency = sparse.coo_matrix((data, (row, col)), shape=(n_row, n_col), dtype=np.int32)
        graph.biadjacency = biadjacency
        graph.names_row = names_row
        graph.names_col = names_col
    
    return graph
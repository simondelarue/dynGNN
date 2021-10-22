#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Oct 20, 2021
@author: Simon Delarue <sdelarue@enst.fr>

This code is heavily inspired from scikit-network package.
'''

import numpy as np
from scipy import sparse

def edgelist2adjacency(edge_list: list) -> sparse.coo_matrix:
    ''' Build an adjacency matrix from a list of edges. The adjacency matrix is stored in COO format.
        
        Parameters
        ----------
        edge_list: list
            List of edges as pairs (i, j) 
            
        Returns
        -------
        adjacency: sparse.coo_matrix

        Exemples
        --------
        >>> edge_list = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 1)]
        >>> adjacency = edgelist2adjacency(edge_list)
        >>> adjacency.shape, adjacency.nnz
        (3, 3) 5
        '''

    edges = np.array(edge_list)
    row, col = edges[:, 0], edges[:, 1]
    n = max(row.max(), col.max()) + 1
    data = np.ones(len(row))

    adjacency = sparse.coo_matrix((data, (row, col)), shape=(n, n), dtype=np.int32)
    
    return adjacency
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Oct 22, 2021
@author: Simon Delarue <sdelarue@enst.fr>
'''

import numpy as np
from scipy import sparse

def get_neighbors(adjacency: sparse.coo_matrix, node: int) -> np.ndarray:
    
    adjacency = adjacency.tocsr()
    neighbors = adjacency.indices[adjacency.indptr[node]: adjacency.indptr[node + 1]]
    return neighbors
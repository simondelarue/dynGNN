#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Oct 22, 2021
@author: Simon Delarue <sdelarue@enst.fr>
'''

import numpy as np
from scipy import sparse

def add_edges(adjacency: sparse.coo_matrix, input_matrix: sparse.coo_matrix) -> sparse.coo_matrix:
    row = np.append(adjacency.row, input_matrix.row)
    col = np.append(adjacency.col, input_matrix.col)
    data = np.append(adjacency.data, input_matrix.data)
    return sparse.coo_matrix((data, (row, col)))
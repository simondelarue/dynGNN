#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Oct 21, 2021
@author: Simon Delarue <sdelarue@enst.fr>
'''

from typing import List

import numpy as np
from scipy import sparse


def split(adjacency: sparse.coo_matrix, batch_size: int) -> List[sparse.coo_matrix]:
    ''' Split COO matrix into chunks, each containing `batch size` number of edges. '''

    row, col, data = adjacency.row, adjacency.col, adjacency.data
    
    row_batches = [row[i:i+batch_size] for i in range(0, len(row), batch_size)]
    col_batches = [col[i:i+batch_size] for i in range(0, len(col), batch_size)]
    data_batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    return [sparse.coo_matrix((data, (row, col)), shape=adjacency.shape, dtype=np.int32) for row, col, data in zip(row_batches, col_batches, data_batches)]
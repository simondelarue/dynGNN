#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Oct 20, 2021
@author: Simon Delarue <sdelarue@enst.fr>

This code is heavily inspired from scikit-network package.
'''

import numpy as np
from scipy import sparse

def bipartite2undirected(biadjacency: sparse.coo_matrix) -> sparse.coo_matrix:
    ''' Adjacency matrix of a bigraph. '''
    
    adjacency = sparse.bmat([[None, biadjacency], [biadjacency.T, None]], format='coo')
    
    return adjacency
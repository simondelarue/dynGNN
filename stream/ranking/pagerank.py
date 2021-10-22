#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Oct 21, 2021
@author: Simon Delarue <sdelarue@enst.fr>

This code is heavily inspired from scikit-network package.
'''

import numpy as np
from scipy import sparse

from stream.ranking.base import BaseRanking
from stream.utils.format import bipartite2undirected

class PageRank(BaseRanking):

    def __init__(self, damping_factor: float = 0.85, solver: str = 'piteration', n_iter: int = 10, tol: float = 1e-6):
        self.damping_factor = damping_factor
        self.solver = solver
        self.n_iter = n_iter 
        self.tol = tol
        self.bipartite = None

    def fit(self, input_matrix: sparse.coo_matrix):
        
        self.row = input_matrix.row
        self.col = input_matrix.col
        self.data = input_matrix.data

        # Format
        n_row, n_col = input_matrix.shape
        if n_row != n_col:
            self.bipartite = True
            adjacency = bipartite2undirected(input_matrix)

        # Get seeds
        seeds_row = np.ones(n_row)
        seeds_col = 0. * np.ones(n_col)
        seeds = np.hstack((seeds_row, seeds_col))
        if seeds.sum() > 0:
            seeds /= seeds.sum()

        # Get pageRank
        self.scores_ = self._get_pagerank(adjacency, seeds, damping_factor=self.damping_factor, n_iter=self.n_iter,
                                        tol=self.tol, solver=self.solver)

        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self

    def update(self, input_matrix: sparse.coo_matrix):
        # update adjacency with new nodes
        self.row = np.append(self.row, input_matrix.row)
        self.col = np.append(self.col, input_matrix.col)
        self.data = np.append(self.data, input_matrix.data)

        # Recompute
        self.fit(sparse.coo_matrix((self.data, (self.row, self.col))))
        
        return self

    def _get_pagerank(self, adjacency: sparse.coo_matrix, seeds: np.ndarray, damping_factor: float, n_iter: int, 
                      tol: float, solver: str = 'piteration') -> np.ndarray:
        ''' PageRank solver. 
            Source : https://asajadi.github.io/fast-pagerank/ '''

        n = adjacency.shape[0]
        
        if solver == 'piteration':

            n = adjacency.shape[0]

            out_degrees = adjacency.dot(np.ones(n)).astype(bool)
            norm = adjacency.dot(np.ones(adjacency.shape[1]))
            diag: sparse.coo_matrix = sparse.diags(norm, format='coo')
            diag.data = 1 / diag.data

            W = (damping_factor * diag.dot(adjacency)).T.tocoo()
            v0 = (np.ones(n) - damping_factor * out_degrees) * seeds
            
            scores = v0

            for i in range(n_iter):
                scores_ = W.dot(scores) + v0 * scores.sum()
                scores_ /= scores_.sum()
                if np.linalg.norm(scores - scores_, ord=1) < tol:
                    break
                else:
                    scores = scores_

        return scores / scores.sum()
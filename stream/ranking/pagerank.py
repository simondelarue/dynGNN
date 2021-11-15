#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Oct 21, 2021
@author: Simon Delarue <sdelarue@enst.fr>

This code is heavily inspired from scikit-network package.
'''

import numpy as np
from scipy import sparse
from scipy.sparse.coo import coo_matrix

from stream.ranking.base import BaseRanking
from stream.utils.format import bipartite2undirected
from stream.utils import get_neighbors, add_edges

class PageRank(BaseRanking):

    def __init__(self, damping_factor: float = 0.85, solver: str = 'piteration', n_iter: int = 10, tol: float = 1e-6):
        self.damping_factor = damping_factor
        self.solver = solver
        self.n_iter = n_iter 
        self.tol = tol
        self.bipartite = None

    def fit(self, input_matrix: sparse.coo_matrix, init_scores: np.ndarray = None):

        # Format
        n_row, n_col = input_matrix.shape
        if n_row != n_col:
            self.bipartite = True
            self.adjacency = bipartite2undirected(input_matrix)
        else:
            self.adjacency = input_matrix

        # Get seeds
        seeds_row = np.ones(n_row)
        if n_row != n_col:
            seeds_col = 0. * np.ones(n_col)
            seeds = np.hstack((seeds_row, seeds_col))
        else:
            seeds = seeds_row

        if seeds.sum() > 0:
            seeds /= seeds.sum()

        # Get pageRank
        self.scores_ = self._get_pagerank(self.adjacency, seeds, damping_factor=self.damping_factor, n_iter=self.n_iter,
                                        tol=self.tol, solver=self.solver, init_scores=init_scores)

        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self

    def update(self, update_matrix: sparse.coo_matrix):
        ''' Perform incremental PageRank ; update PageRank scores for every nodes after updating
            adjacency matrix. '''

        # Keep track of previous step scores
        prev_scores = self.scores_.copy()

        # update adjacency with new nodes
        if self.bipartite:
            update_adjacency = bipartite2undirected(update_matrix)
        else:
            update_adjacency = update_matrix

        updated_adjacency = add_edges(self.adjacency, update_adjacency)


        # Compute PageRank scores on changed nodes
        # Get seeds
        n_row, n_col = updated_adjacency.shape
        seeds_row = np.ones(n_row)
        if n_row != n_col:
            seeds_col = 0. * np.ones(n_col)
            seeds = np.hstack((seeds_row, seeds_col))
        else:
            seeds = seeds_row

        if seeds.sum() > 0:
            seeds /= seeds.sum()

        scores = self._get_pagerank(updated_adjacency, seeds, damping_factor=self.damping_factor, n_iter=self.n_iter,
                                        tol=self.tol, solver=self.solver, init_scores=prev_scores)

        self.scores_ = scores.copy()

        # Keep track of updated_adj adjacency
        self.adjacency = updated_adjacency

        if self.bipartite:
            self._split_vars(update_matrix.shape)

        return self
        

    def update_selected(self, update_matrix: sparse.coo_matrix):
        ''' Perform incremental PageRank ; update PageRank scores only for nodes that changed compared to previous step, 
            i.e nodes on which new edge is attached. '''
        
        # Keep track of previous step scores
        prev_scores = self.scores_.copy()

        # update adjacency with new nodes
        if self.bipartite:
            update_adjacency = bipartite2undirected(update_matrix)
        else:
            update_adjacency = update_matrix

        updated_adjacency = add_edges(self.adjacency, update_adjacency)

        # Select changed nodes
        changed_adjacency = (updated_adjacency - self.adjacency).tocoo() # TODO How to consider weights in adjacency matrix ? 

        # Compute PageRank scores on changed nodes
        # Get seeds
        n_row, n_col = changed_adjacency.shape
        seeds_row = np.ones(n_row)
        if n_row != n_col:
            seeds_col = 0. * np.ones(n_col)
            seeds = np.hstack((seeds_row, seeds_col))
        else:
            seeds = seeds_row

        if seeds.sum() > 0:
            seeds /= seeds.sum()

        scores = self._get_pagerank(changed_adjacency, seeds, damping_factor=self.damping_factor, n_iter=self.n_iter,
                                        tol=self.tol, solver=self.solver, init_scores=prev_scores)

        # Update PageRank scores for changed nodes
        idx_changed_nodes = np.array(list(set(np.nonzero(changed_adjacency)[0]).union(set(np.nonzero(changed_adjacency)[1]))))
        #print(f'Index changed nodes : {len(idx_changed_nodes)}')

        prev_scores[idx_changed_nodes] = scores[idx_changed_nodes]
        self.scores_ = prev_scores.copy()

        # Keep track of updated_adj adjacency
        self.adjacency = updated_adjacency

        if self.bipartite:
            self._split_vars(update_matrix.shape)

        return self

    def _get_pagerank(self, adjacency: sparse.coo_matrix, seeds: np.ndarray, damping_factor: float, n_iter: int, 
                      tol: float, solver: str = 'piteration', init_scores: np.ndarray = None) -> np.ndarray:
        ''' PageRank solver '''

        n = adjacency.shape[0]
        
        if solver == 'piteration':

            out_degrees = adjacency.dot(np.ones(n)).astype(bool)
            norm = adjacency.dot(np.ones(adjacency.shape[1]))
            diag: sparse.coo_matrix = sparse.diags(norm, format='coo')
            diag.data = 1 / diag.data

            W = (damping_factor * diag.dot(adjacency)).T.tocoo()
            v0 = (np.ones(n) - damping_factor * out_degrees) * seeds
            
            if init_scores is not None:
                scores = init_scores
            else:
                scores = v0

            for i in range(n_iter):
                scores_ = W.dot(scores) + v0 * scores.sum()
                scores_ /= scores_.sum()
                if np.linalg.norm(scores - scores_, ord=1) < tol:
                    break
                else:
                    scores = scores_

        return scores / scores.sum()
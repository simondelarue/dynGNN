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
        
        self.row = input_matrix.row
        self.col = input_matrix.col
        self.data = input_matrix.data

        # Format
        n_row, n_col = input_matrix.shape
        if n_row != n_col:
            self.bipartite = True
            adjacency = bipartite2undirected(input_matrix)
            self.adjacency_init = adjacency.copy()
        else:
            adjacency = input_matrix

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
        self.scores_ = self._get_pagerank(adjacency, seeds, damping_factor=self.damping_factor, n_iter=self.n_iter,
                                        tol=self.tol, solver=self.solver, init_scores=init_scores)

        if self.bipartite:
            self._split_vars(input_matrix.shape)

        return self

    def update(self, input_matrix: sparse.coo_matrix):

        prev_scores = self.scores_

        # update adjacency with new nodes
        n_row, n_col = input_matrix.shape
        if n_row != n_col:
            input_matrix = bipartite2undirected(input_matrix)
        
        adjacency_tot = add_edges(self.adjacency_init, input_matrix)
        
        # Incremental PageRank
        # Build sets v_u(nchanged) and v_c(hanged)
        nodes = set(self.row).union(set(self.col))
        next_nodes = set(input_matrix.row).union(set(input_matrix.col))

        v_c = next_nodes.copy()
        v_u = nodes - v_c

        # Step 1 - Iniitalize v_q
        v_q = set()

        # Step 2 - Extend changed vertices to include all descendants of elements in v_c
        while len(v_c) != 0:
            n = v_c.pop()
            v_q.add(n)

            #print(f'Adjacency shape : {adjacency_tot.shape} - n : {n}')
            for neighb in get_neighbors(adjacency_tot, n):
                if neighb in v_u:
                    v_u.discard(neighb)
                    v_c.add(neighb)

        # Step 3 - Scale elements in unchanged list v_u
        v_b = set()

        for n_u in v_u:
            prev_scores[n_u] = prev_scores[n_u] * (len(v_u) / len(next_nodes))

            for neighb_u in get_neighbors(adjacency_tot, n_u):
                if neighb_u in v_q:
                    v_u.pop(neighb_u)
                    v_b.add(neighb_u)

        # Step 4 - Scale border nodes + PageRank on changed nodes
        for n_b in v_b:
            prev_scores[n_b] = prev_scores[n_b] * (len(v_u) / len(v_q))

        #self.scores_ = self.fit(sparse.coo_matrix((self.data, (self.row, self.col))))
        # Changed nodes adjacency matrix
        n_c_all = np.array(list(v_q.union(v_b)))
        
        # remove edges from coo_matrix
        adjacency_c = adjacency_tot.copy()
        for i, j, d in zip(adjacency_tot.row, adjacency_tot.col, adjacency_tot.data):
            if i in n_c_all:
                mask = adjacency_c.row == i
                adjacency_c.row = adjacency_c.row[mask]
                adjacency_c.col = adjacency_c.col[mask]
                adjacency_c.data = adjacency_c.data[mask]
            elif j in n_c_all:
                mask = adjacency_c.col == j
                adjacency_c.col = adjacency_c.col[mask]
                adjacency_c.row = adjacency_c.row[mask]
                adjacency_c.data = adjacency_c.data[mask]

        adjacency_c = sparse.coo_matrix((adjacency_c.data, (adjacency_c.row, adjacency_c.col)))
        
        self.fit(adjacency_c, init_scores=prev_scores)

        # Update and save pagerank result for nodes that changed 
        prev_scores[n_c_all] = self.scores_
        self.scores_ = prev_scores

        return self

    def _get_pagerank(self, adjacency: sparse.coo_matrix, seeds: np.ndarray, damping_factor: float, n_iter: int, 
                      tol: float, solver: str = 'piteration', init_scores: np.ndarray = None) -> np.ndarray:
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
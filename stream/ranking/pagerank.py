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

        prev_scores = self.scores_

        # update adjacency with new nodes
        n_row, n_col = update_matrix.shape
        if n_row != n_col:
            update_matrix = bipartite2undirected(update_matrix)
        
        updated_adj = add_edges(self.adjacency, update_matrix)
        
        # Incremental PageRank
        # Build sets v_u(nchanged) and v_c(hanged)
        nodes = set(self.adjacency.row).union(set(self.adjacency.col))
        input_nodes = set(update_matrix.row).union(set(update_matrix.col))

        v_c = input_nodes.copy()
        v_u = nodes.difference(v_c)

        # Step 1 - Iniitalize v_q
        v_q = set()

        # Step 2 - Extend changed vertices to include all descendants of elements in v_c
        while len(v_c) != 0:
            n = v_c.pop()
            v_q.add(n)

            #print(f'Adjacency shape : {adjacency_tot.shape} - n : {n}')
            for neighb in get_neighbors(updated_adj, n):
                if neighb in v_u:
                    v_u.discard(neighb)
                    v_c.add(neighb)

        # Step 3 - Scale elements in unchanged list v_u
        v_b = set()

        for n_u in v_u:
            prev_scores[n_u] = prev_scores[n_u] * (len(v_u) / len(input_nodes))

            for neighb_u in get_neighbors(updated_adj, n_u):
                if neighb_u in v_q:
                    v_u.discard(neighb_u)
                    v_b.add(neighb_u)

        # Step 4 - Scale border nodes + PageRank on changed nodes
        for n_b in v_b:
            prev_scores[n_b] = prev_scores[n_b] * (len(v_u) / len(v_q))

        #self.scores_ = self.fit(sparse.coo_matrix((self.data, (self.row, self.col))))
        # Changed nodes adjacency matrix
        n_c_all = v_q.union(v_b)
        
        # PageRank is computed only on new nodes regarding previous batch
        rows_c, cols_c, data_c = np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        for i, j, d in zip(updated_adj.row, updated_adj.col, updated_adj.data):
            if i in n_c_all:
                mask = updated_adj.row == i
                rows_c = np.append(rows_c, updated_adj.row[mask])
                cols_c = np.append(cols_c, updated_adj.col[mask])
                data_c = np.append(data_c, updated_adj.data[mask])
                n_c_all.discard(i)
            elif j in n_c_all:
                mask = updated_adj.col == j
                cols_c = np.append(cols_c, updated_adj.col[mask])
                rows_c = np.append(rows_c, updated_adj.row[mask])
                data_c = np.append(data_c, updated_adj.data[mask])
                n_c_all.discard(j)

        adjacency_c = sparse.coo_matrix((data_c, (rows_c, cols_c)), shape=self.adjacency.shape, dtype=np.int32)
        
        self.fit(adjacency_c, init_scores=prev_scores)

        # Update and save pagerank result for nodes that changed 
        prev_scores = self.scores_
        self.scores_ = prev_scores

        return self

    def update_naive(self, update_matrix: sparse.coo_matrix):
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
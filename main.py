# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg.linalg import norm
from scipy import sparse

from stream.utils import edgelist2adjacency
from stream.data import load_data, split
from stream.ranking import PageRank, top_k, MRR

#import sknetwork as skn
#from sknetwork.data import movie_actor
#from sknetwork.ranking import PageRank as skPageRank
#from sknetwork.linalg.normalization import normalize

import matplotlib.pyplot as plt

from line_profiler import LineProfiler

#def run_skpr(adj):
#   skpagerank = skPageRank()
#    scores_sk = skpagerank.fit_transform(adj)

def run_pr(adj):
    pagerank = PageRank()
    scores = pagerank.fit_transform(adj)


if __name__=='__main__':

    # From numpy array
    adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    adj = sparse.coo_matrix(adj)
    '''print(adj)
    print(adj.shape, adj.nnz)
    print(adj.todense())
    print('\n')'''

    # From edge list
    edge_list = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 1)]
    adj = edgelist2adjacency(edge_list)
    '''print(adj)
    print(adj.todense())
    print(adj.shape, adj.nnz)'''

    # Load data
    graph = load_data('ml-latest-small', data_home='streamdata')
    print('Row, col, data lengths : ', len(graph.biadjacency.row), len(graph.biadjacency.col), len(graph.biadjacency.data))
    print(f'Adjacency shape : {graph.biadjacency.shape}')

    # Split data
    adj_batches = split(graph.biadjacency, batch_size=20000)
    print(f'Number of batches : {len(adj_batches)}')
    '''for i in adj_batches:
        print(i.nnz)'''
    
    # Static PageRank

    # MovieLens with coo PageRank
    '''lp = LineProfiler()
    lp_wrapper = lp(run_pr)
    lp_wrapper(graph.biadjacency)
    lp.print_stats()'''
    
    # MovieLens with csr (sknetwork) PageRank
    '''lp = LineProfiler()
    lp_wrapper = lp(run_skpr)
    lp_wrapper(graph.biadjacency.tocsr())
    lp.print_stats()'''

    # PageRank on movie_acctor
    '''
    graph = movie_actor(metadata=True)
    biadjacency = graph.biadjacency
    names_row = graph.names_row
    names_col = graph.names_col
    
    skpagerank = skPageRank()
    skpagerank.fit(biadjacency)
    scores_row = skpagerank.scores_row_
    scores_col = skpagerank.scores_col_
    print('Scores row : ', scores_row)
    print('Scores col : ', scores_col)

    # My PageRank
    pagerank = PageRank()
    pagerank.fit(biadjacency.tocoo())
    scores_row = pagerank.scores_row_
    scores_col = pagerank.scores_col_
    print('Scores row : ', scores_row)
    print('Scores col : ', scores_col)
    '''
    static_pr = PageRank()
    static_scores = static_pr.fit_transform(graph.biadjacency)
    static_scores_row = static_pr.scores_row_
    static_scores_col = static_pr.scores_col_

    print(f'\nStatic top PR nodes row : {top_k(static_scores_row, 10)}')
    print(f'Static top PR scores row : {np.round(sorted(static_scores_row, reverse=True)[:10], 4)}')
    print(f'Static top PR nodes col : {top_k(static_scores_col, 10)}')
    print(f'Static top PR scores col : {np.round(sorted(static_scores_col, reverse=True)[:10], 4)}')

    # Dynamic PageRank
    pagerank = PageRank()
    dyn_scores_row = [] 
    dyn_scores_col = []
    scores = pagerank.fit_transform(adj_batches[0])
    scores_row = pagerank.scores_row_
    scores_col = pagerank.scores_col_
    dyn_scores_row.append(scores_row)
    dyn_scores_col.append(scores_col)
    print('\nInit increm nodes row  : ', top_k(scores_row, 10))
    print('Init increm scores row : ', sorted(np.round(scores_row, 4), reverse=True)[:10])
    print('Init increm nodes col  : ', top_k(scores_col, 10))
    print('Init increm scores col : ', sorted(np.round(scores_col, 4), reverse=True)[:10])

    for idx, batch in enumerate(adj_batches[1:]):
        scores = pagerank.update_transform(batch)
        scores_row = pagerank.scores_row_.copy()
        scores_col = pagerank.scores_col_.copy()

        # Update list
        dyn_scores_row.append(scores_row)
        dyn_scores_col.append(scores_col)

    print('\nFinal increm nodes row  : ', top_k(dyn_scores_row[-1], 10))
    print('Final increm scores row : ', sorted(np.round(dyn_scores_row[-1], 4), reverse=True)[:10])
    print('Final increm nodes col  : ', top_k(dyn_scores_col[-1], 10))
    print('Final increm scores col : ', sorted(np.round(dyn_scores_col[-1], 4), reverse=True)[:10])

    nb_found_n_row = len(set(top_k(dyn_scores_row[-1], 100)).intersection(set(top_k(static_scores_row, 100))))
    print(f'\nPercentage of nodes row found in top 100 (not considering position) : {nb_found_n_row}')
    nb_found_n_col = len(set(top_k(dyn_scores_col[-1], 100)).intersection(set(top_k(static_scores_col, 100))))
    print(f'Percentage of nodes row found in top 100 (not considering position) : {nb_found_n_col}')

    # Computes Mean Reciprocal Rank
    mrr_row = MRR(static_scores_row, dyn_scores_row[-1], k=5)
    mrr_col = MRR(static_scores_col, dyn_scores_col[-1], k=5)
    print(f'Mean Reciprocal Rank for rows and cols : {mrr_row:.3f} - {mrr_col:.3f}')


    def split_data(graph, batch_size):
        # Split data
        adj_batches = split(graph.biadjacency, batch_size=batch_size)

        return adj_batches

    def incremental_PageRank(batches):
        # Incremental PageRank
        pagerank = PageRank()
        dyn_scores_row, dyn_scores_col = [], []
        scores = pagerank.fit_transform(batches[0])
        scores_row = pagerank.scores_row_
        scores_col = pagerank.scores_col_
        
        dyn_scores_row.append(scores_row)
        dyn_scores_col.append(scores_col)

        # Iterate through all batches of edges
        for idx, batch in enumerate(batches[1:]):
            scores = pagerank.update_transform(batch)
            scores_row = pagerank.scores_row_.copy()
            scores_col = pagerank.scores_col_.copy()
            dyn_scores_row.append(scores_row)
            dyn_scores_col.append(scores_col)

        return dyn_scores_row, dyn_scores_col


    # Analysis of pics
    k_vals = [5]
    batch_sizes = [100836]

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    colors = ['black', 'green', 'blue', 'red']

    for ax_idx, batch_size in enumerate(batch_sizes):
        batches = split_data(graph, batch_size)
        dyn_scores_row, dyn_scores_col = incremental_PageRank(batches)
        
        # Plots
        for i, k in enumerate(k_vals):
            X = range(len(dyn_scores_row))
            y = [MRR(static_scores_row, batch_r, k=k) for batch_r in dyn_scores_row]
            plt.plot(X, y, label=f'k={k} (row)', color=colors[i], marker='+')
            plt.text(max(X)+0.5, y[-1]+0.01, f'{y[-1]:.2f}')
            #plt.plot(range(len(dyn_scores_col)), 
            #        [MRR(static_scores_col, batch_c, k=k) for batch_c in dyn_scores_col], 
            #        label=f'k={k} (col)', color=colors[i], marker='*', alpha=0.3)
        plt.xlabel('Batches')
        plt.ylabel('MRR')
        plt.legend()
        plt.title(f'Evolution of MRR over batches of edges, according to value of $k$ (batch size={batch_size})', weight='bold'); 



    
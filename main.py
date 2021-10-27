# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg.linalg import norm
from scipy import sparse

from stream.utils import edgelist2adjacency
from stream.data import load_data, split
from stream.ranking import PageRank, pagerank

#import sknetwork as skn
#from sknetwork.data import movie_actor
#from sknetwork.ranking import PageRank as skPageRank
#from sknetwork.linalg.normalization import normalize

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
    adj_batches = split(graph.biadjacency, batch_size=10000)
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
    
    def top_k(scores, n):
        return np.argpartition(scores, -n)[-n:]

    print(f'\nStatic top PR nodes : {top_k(static_scores, 10)}')
    print(f'\nStatic top PR scores : {sorted(static_scores, reverse=True)[:10]}')

    # Dynamic PageRank
    pagerank = PageRank()
    dyn_scores = []
    scores = pagerank.fit_transform(adj_batches[0])
    dyn_scores.append(scores)
    print('\nInit increm nodes  : ', top_k(scores, 10))
    print('\nInit increm scores : ', sorted(scores, reverse=True)[:10])

    for idx, batch in enumerate(adj_batches[1:100]):
        scores = pagerank.update_transform(batch)
        dyn_scores.append(scores)
        #print(f'\nIncremental scores batch {idx+1} : ', [round(x, 4) for x in sorted(scores, reverse=True)[:10]])
    
    print(f'\nDynamic PR nodes : {top_k(dyn_scores[-1], 10)}')
    print(f'\nDynamic PR scores : {sorted(dyn_scores[-1], reverse=True)[:10]}')

    nb_found_n = len(set(top_k(dyn_scores[-1], 100)).intersection(set(top_k(static_scores, 100))))
    print(f'\nPercentage of nodes found in top 100 (not considering position) : {nb_found_n}')











    
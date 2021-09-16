import numpy as np
from scipy import sparse
from scipy.sparse.csr import csr_matrix
import time
from collections import defaultdict
import pandas as pd
from mydict import MyDict


def csr2dict(csr_matrix):
    ''' For each row in adjacency matrix and if the row does contain non zero values,
        we retrieve columns where values are non zeros, and add corresponding data to adjacency dictionary '''
    
    # Initalize adjacency as a dictionary (of dictionary)
    adj = {}
    
    # Fill adjacency dictionary
    for i in range(1, len(csr_matrix.indptr)-1):
        if (csr_matrix.indptr[i] - csr_matrix.indptr[i-1]>0):
            columns = csr_matrix.indices[csr_matrix.indptr[i-1]:csr_matrix.indptr[i]]
            data = csr_matrix.data[csr_matrix.indptr[i-1]:csr_matrix.indptr[i]]
            adj[i-1] = {col: val for col, val in zip(columns, data)}
            
    return adj


def create_sparse_matrix(n, density):
    m = n
    size = int(n * m * density)
    rows = np.random.randint(0, n, size=size)
    cols = np.random.randint(0, m, size=size)
    data = np.random.randint(1, 2, size)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, m))


def vecmul(adj, v):
    ''' Matrix-vector product with dictionaries. '''
    res = {}
    for key, value in adj.items():
        for key_col, value_col in value.items():
            res[key] = res.get(key, 0) + value_col*v[key_col]
    return res


def run(ns, density, method):

    history = defaultdict(list)
    for n in ns:
        for i in range(3):
            nb_nodes = int(n)
            # CSR adjacency
            A = create_sparse_matrix(nb_nodes, density)
            # Random vector
            v = {k: v for k, v in enumerate(np.random.rand(A.shape[0]))}
            # Dictionary adjacency
            adj = csr2dict(A)

            if method == 'numpy':
                # Dot-product with Numpy
                v = np.random.randn(A.shape[0])
                start = time.time()
                res = A.dot(v)
                end = time.time()
                history[n].append(end-start)

            if method == 'python_dict':
                # Dot-product with python dict
                start = time.time()
                res = vecmul(adj, v)
                end = time.time()
                history[n].append(end-start)
            
            elif method == 'cython_dict':
                my_dict = MyDict(A)
                my_v = MyDict(v)
                # Dot-product with cython dict
                start = time.time()
                res = my_dict.dot(my_v)
                end = time.time()
                history[n].append(end-start)

    return history


def print_history(history, method):

    print(f'------ {method} -------')
    for key, value in history.items():
        print(f'{key} : {np.mean(value):.4f}')


if __name__=='__main__':

    # Parameters
    ns = [1e3, 1e4, 1e5]
    density = 0.001

    # Dot-products
    methods = ['numpy', 'python_dict', 'cython_dict']
    hist_list = []

    for method in methods:
        hist = run(ns, density, method)
        print_history(hist, method)
        df = pd.melt(pd.DataFrame(hist))
        df['method'] = method
        hist_list.append(df)

    # Transform results for plotting
    df_plot = pd.concat(hist_list)
    df_plot.to_pickle('res_plot.pkl')
    

    # ----- myDict
    #A_dense = np.array([[0, 0, 0, 0, 0],
    #                    [0, 0, 0, 1, 1],
    #                    [0, 0, 1, 0, 0],
    #                    [1, 0, 1, 0, 0],
    #                    [0, 0, 0, 0, 0]])
    #A = csr_matrix(A_dense)
    #my_dict = MyDict(A)
    #print(f'my dict : {my_dict}')

    # ----- myVect (random)
    #v = {k: v for k, v in enumerate(np.random.rand(A.shape[0]))}
    #v = {0: 0.5, 1: 1, 2: 0.5, 3: 2, 4: 1}
    #my_v = MyDict(v)
    #print('My random v : ', my_v)

    # ----- Dot-product
    # Note : random vector v can either be a dictionary or a MyDict object 
    # (if the latter is true, it is the associated unordered map of the object
    # which is considered when processing dot product).
    #res = my_dict.dot(my_v)
    #print(f'Result = {res}')
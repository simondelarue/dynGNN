import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.csr import csr_matrix
from scipy.sparse.coo import coo_matrix
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
    for i in range(1, len(csr_matrix.indptr)):
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
    v_len = len(v)
    for key, value in adj.items():
        tmp = 0
        # Iterates over smallest set of values between columns in A and values in v
        if len(value) < v_len:
            for key_col, value_col in value.items():
                #res[key] = res.get(key, 0) + value_col*v.get(key_col, 0)
                tmp += value_col * v.get(key_col, 0)
        else:
            for vec_col, vec_value in v.items():
                #res[key] = res.get(key, 0) + value.get(vec_col, 0) * vec_value
                tmp += value.get(vec_col, 0) * vec_value
        if tmp != 0:
            res[key] = tmp
    return res


def run_test(method, A, v, v_dict):

    # Dictionary adjacency
    adj = csr2dict(A)

    if method == 'CSR':
        # Dot-product with Numpy
        res = A.dot(v)

    elif method == 'python_dict':
        # Dot-product with python dict
        res = vecmul(adj, v_dict)
        res = np.sum([v for _, v in res.items()])
    
    elif method == 'cython_dict':
        my_dict = MyDict(A)
        my_v = MyDict(v_dict)
        # Dot-product with cython dict
        res = my_dict.dot(my_v)
        res = np.sum([v for _, v in res.items()])

    elif method == 'cython_dict_map':
        my_dict = MyDict(A)
        my_v = MyDict(v_dict)
        # Dot-product with cython dict
        res = my_dict.dot_map(my_v)
        res = np.sum([v for _, v in res.items()])

    elif method == 'cython_dict_opt':
        my_dict = MyDict(A)
        my_v = MyDict(v_dict)
        # Dot-product with cython dict
        res = my_dict.dot_map(my_v)
        res = np.sum([v for _, v in res.items()])

    elif method == 'COO':
        A_coo = A.tocoo()
        # Dot-product with cython dict
        res = A_coo.dot(v)
    
    elif method == 'CSC':
        A_csc = A.tocsc()
        # Dot-product with cython dict
        res = A_csc.dot(v)

    elif method == 'LIL':
        A_lil = A.tolil()
        # Dot-product with cython dict
        res = A_lil.dot(v)
    
    return res

if __name__=='__main__':

    # ===== RUN OVER SET OF PARAMS ======
    # Parameters
    ns = [1e3, 1e4, 1e5]#, 1e6]
    densities = [1e-2, 1e-3, 1e-4]
    dense_v = True

    # Dot-products
    methods = ['CSR', 'CSC', 'COO', 'LIL', 'cython_dict', 'cython_dict_map', 'python_dict']

    for n in ns:
        for density in densities:
            print('\n=========================================================')
            nb_nodes = int(n)
            # CSR adjacency
            A = create_sparse_matrix(nb_nodes, density)
            # Random vector
            if dense_v:
                v = np.random.randn(A.shape[0])
                #v = np.random.randint(0, 2, A.shape[0])
                v_dict = {k: v for k, v in enumerate(v) if v!=0}
            else:
                v = scipy.sparse.random(A.shape[0], 1, density=density)
                v_dict = {k: v[0, 0] for k, v in enumerate(v.todense()) if v!=0}
            for method in methods:
                res = run_test(method, A, v, v_dict)
                print(f'n:{n:>10} - density:{density:>6} - method:{method:>16} | res:{res.sum().sum()}')
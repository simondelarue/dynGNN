import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.csr import csr_matrix
from scipy.sparse.coo import coo_matrix
import time
from collections import defaultdict
import pandas as pd
import random
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


def run(ns, densities, method, dense_v=True):

    history_tot = {}
    for n in ns:
        history = defaultdict(list)
        for density in densities:
            for i in range(3):
                nb_nodes = int(n)
                # CSR adjacency
                A = create_sparse_matrix(nb_nodes, density)
                # Random vector
                if dense_v:
                    v = np.random.randn(A.shape[0])
                    v_dict = {k: v for k, v in enumerate(v) if v!=0}
                else:
                    v = scipy.sparse.random(A.shape[0], 1, density=density)
                    v_dict = {k: v[0, 0] for k, v in enumerate(v.todense()) if v!=0}
                # Dictionary adjacency
                adj = csr2dict(A)

                if method == 'CSR':
                    # Dot-product with Numpy
                    start = time.time()
                    res = A.dot(v)
                    end = time.time()
                    history[density].append(end-start)

                if method == 'python_dict':
                    # Dot-product with python dict
                    start = time.time()
                    res = vecmul(adj, v_dict)
                    end = time.time()
                    history[density].append(end-start)
                
                elif method == 'cython_dict':
                    my_dict = MyDict(A)
                    my_v = MyDict(v_dict)
                    # Dot-product with cython dict
                    start = time.time()
                    res = my_dict.dot(my_v)
                    end = time.time()
                    history[density].append(end-start)

                elif method == 'cython_dict_map':
                    my_dict = MyDict(A)
                    my_v = MyDict(v_dict)
                    # Dot-product with cython dict
                    start = time.time()
                    res = my_dict.dot_map(my_v)
                    end = time.time()
                    history[density].append(end-start)

                elif method == 'cython_dict_opt':
                    my_dict = MyDict(A)
                    my_v = MyDict(v_dict)
                    # Dot-product with cython dict
                    start = time.time()
                    res = my_dict.dot_map(my_v)
                    end = time.time()
                    history[density].append(end-start)

                elif method == 'COO':
                    A_coo = A.tocoo()
                    # Dot-product 
                    start = time.time()
                    res = A_coo.dot(v)
                    end = time.time()
                    history[density].append(end-start)

                elif method == 'COO_shuffled':
                    A_coo = A.tocoo()
                    row = A_coo.row
                    col = A_coo.col
                    data = A_coo.data
                    mask = np.random.choice(np.arange(0, len(row)), size=len(row), replace=False)
                    # Shuffle coo matrix
                    row = row[mask]
                    col = col[mask]
                    data = data[mask]
                    A_coo = coo_matrix((data, (row, col)))
                    # Reshape v according to COO matrix shape
                    v = np.array(v.todense()).ravel()[:A_coo.shape[1]]
                    # Dot product
                    start = time.time()
                    res = A_coo.dot(v)
                    end = time.time()
                    history[density].append(end-start)

                elif method == 'CSC':
                    A_csc = A.tocsc()
                    # Dot-product 
                    start = time.time()
                    res = A_csc.dot(v)
                    end = time.time()
                    history[density].append(end-start)

                elif method == 'LIL':
                    A_lil = A.tolil()
                    # Dot-product 
                    start = time.time()
                    res = A_lil.dot(v)
                    end = time.time()
                    history[density].append(end-start)

                elif method == 'DOK':
                    A_dok = A.todok()
                    # Dot-product 
                    start = time.time()
                    res = A_dok.dot(v)
                    end = time.time()
                    history[density].append(end-start)

        history_tot[n] = history
    return history_tot

def run_toy(method='CSR', dense_v=True):

    print(f' ==== TOY EXAMPLE ({method}) === ')
    # ----- myDict
    A_dense = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1],
                        [0, 0, 1, 0, 0],
                        [1, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0]])
    A = csr_matrix(A_dense)
    if method=='cython_dict':
        A = MyDict(A)
        print(f'my dict : {A}')
    elif method=='python_dict':
        A = csr2dict(A)
    elif method=='COO':
        A = coo_matrix(A_dense)
    elif method == 'COO_shuffled':
        A = coo_matrix(A_dense)
        row = A.row
        col = A.col
        data = A.data
        mask = np.random.choice(np.arange(0, len(row)), size=len(row), replace=False)
        # Shuffle coo matrix
        row = row[mask]
        col = col[mask]
        data = data[mask]
        A = coo_matrix((data, (row, col)))


    # ----- myVect (random)
    if dense_v:
        my_v = {0: 0.5, 1: 1, 2: 0.5, 3: 2, 4: 1}
    else:
        my_v = {2: 1}

    if method=='cython_dict':
        my_v = MyDict(my_v)
        print('My random v : ', my_v)
    elif method in ['COO', 'COO_shuffled']:
        if dense_v:
            my_v = coo_matrix(np.array([0.5, 1, 0.5, 2, 1])).T
        else:
            my_v = coo_matrix(np.array([0, 0, 1, 0, 0])).T

    # ----- Dot-product
    # Note : random vector v can either be a dictionary or a MyDict object 
    # (if the latter is true, it is the associated unordered map of the object
    # which is considered when processing dot product).
    if method=='python_dict':
        res = vecmul(A, my_v)
    else:
        res = A.dot(my_v)

    print(f'Result = {res}')
    if dense_v:
        print('Correct result : {3: 1.0, 2: 0.5, 1: 3.0}')
    else:
        print('Correct result : {2: 1.0, 3: 1.0}')


def print_history(history, method):
    print(f'------ {method} -------')
    for key, values in history.items():
        for key_d, value_d in values.items():
            print(f'N={key} - Density={key_d} : {np.mean(value_d):.4f}')


if __name__=='__main__':

    # ===== TOY EXAMPLE ======
    '''dense_v = True
    run_toy(method='COO_shuffled', dense_v=dense_v)
    run_toy(method='cython_dict', dense_v=dense_v)
    run_toy(method='python_dict', dense_v=dense_v)'''

    # ===== RUN OVER SET OF PARAMS ======
    # Parameters
    ns = [1e3, 1e4, 1e5, 1e6]
    #ns = [1e-5, 1e6]
    #densities = [1e-3, 1e-4, 1e-5]
    densities = [1e-4, 1e-5, 1e-6]
    dense_v = False

    # Dot-products
    #methods = ['CSR', 'CSC', 'COO', 'LIL', 'DOK', 'cython_dict', 'cython_dict_map', 'python_dict']
    methods = ['COO_shuffled', 'CSR', 'CSC', 'COO', 'python_dict', 'cython_dict']
    hist_list = []

    for method in methods:
        hist = run(ns, densities, method, dense_v=dense_v)
        print_history(hist, method)
        
        # convert to DataFrame
        for k, values in hist.items():
            df_tmp = pd.melt(pd.DataFrame(values))
            df_tmp['N'] = k
            df_tmp['method'] = method
            hist_list.append(df_tmp)

    # Transform results for plotting
    df_plot = pd.concat(hist_list)
    df_plot.to_pickle('results/res_plot_full_sparse_v_shuff.pkl', protocol=3)
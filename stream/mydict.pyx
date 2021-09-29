# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import numpy as np
from typing import Union
from scipy import sparse
from cython.operator cimport dereference, postincrement
from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map as cpp_map
from libcpp.vector cimport vector
import time

class MyDict():

    # Original code
    def __init__(self, input: Union[sparse.csr_matrix, dict]):
        if type(input)==dict:
            self.my_dict = self.__dict2map(input)
        else:
            self.my_dict = self.__csr2map(input)

    def dot(self, v):
        start = time.time()
        # Inputs
        cdef unordered_map[int, float] v_map
        cdef unordered_map[int, unordered_map[int, float]] A #= self.my_dict
        cdef unordered_map[int, float] values
        # Iterators
        cdef unordered_map[int, unordered_map[int, float]].iterator keys_it 
        cdef unordered_map[int, float].iterator values_it
        cdef unordered_map[int, float].iterator values_v_it
        # Result
        cdef unordered_map[int, float] res
        cdef float tmp
        end = time.time()
        #print(f'    Initialization time : {(end-start):.7f}s')

        start = time.time()
        if isinstance(v, dict):
            v_map = self.__dict2map(v)
        elif isinstance(v, MyDict):
            v_map = v.my_dict
        #print('type : ', type(self.my_dict))
        A = self.my_dict # This cast operation takes most of the computation time !
        keys_it = A.begin()
        end = time.time()
        #print(f'    Cast time : {(end-start):.7f}s')
        start = time.time()

        # Iterates over keys in left matrix
        while(keys_it != A.end()):
            tmp = 0.0
            # Iterates over smallest set of values between A and v
            if A.size() > dereference(keys_it).second.size():
                values_it = dereference(keys_it).second.begin()
                while(values_it != dereference(keys_it).second.end()):
                    # Computes dot-product
                    tmp += dereference(values_it).second * v_map[dereference(values_it).first]
                    postincrement(values_it)
            else:
                values_v_it = v_map.begin()
                values = dereference(keys_it).second
                while(values_v_it != v_map.end()):
                    # Computes dot-product
                    tmp += dereference(values_v_it).second * values[dereference(values_v_it).first]
                    postincrement(values_v_it)

            # Update result
            if tmp != 0.0:
                res[dereference(keys_it).first] = tmp
            postincrement(keys_it)
        end = time.time()
        #print(f'   Loop time : {(end-start):.7f}s')

        return res

    def dot_opt(self, v):
        start = time.time()
        # Inputs
        cdef unordered_map[int, float] v_map
        cdef unordered_map[int, unordered_map[int, float]] A #= self.my_dict
        # Iterators
        cdef unordered_map[int, unordered_map[int, float]].iterator keys_it 
        cdef unordered_map[int, float].iterator values_it
        # Result
        cdef unordered_map[int, float] res
        cdef float tmp
        end = time.time()
        print(f'    Initialization time : {(end-start):.7f}s')

        start = time.time()
        if isinstance(v, dict):
            v_map = self.__dict2map(v)
        elif isinstance(v, MyDict):
            v_map = v.my_dict
        print('type : ', type(self.my_dict))
        A = self.my_dict # This cast operation takes most of the computation time !
        keys_it = A.begin()
        end = time.time()
        print(f'    Cast time : {(end-start):.7f}s')

        start = time.time()
        # Iterates over keys in left matrix
        for key_r, values in A:
            print(key_r)
            print(values)
            tmp = 0.0
            print(type(values))
            print(values.items())
            #for key_col, value_col in values:
            #    print(key_col)
            #    print(value_col)
            #    tmp += value_col * v_map[key_col]
            #res[key_r] = tmp
        end = time.time()
        print(f'   Loop time : {(end-start):.7f}s')
        
        return res

    def dot_map(self, v):

        cdef cpp_map[int, float] v_map
        cdef cpp_map[int, cpp_map[int, float]] A = self.my_dict
        cdef cpp_map[int, float] values
        # Iterators
        cdef cpp_map[int, cpp_map[int, float]].iterator keys_it 
        cdef cpp_map[int, float].iterator values_it
        cdef cpp_map[int, float].iterator values_v_it
        # Result
        cdef cpp_map[int, float] res
        cdef float tmp

        if isinstance(v, dict):
            v_map = self.__dict2map(v)
        elif isinstance(v, MyDict):
            v_map = v.my_dict

        A = self.my_dict # This cast operation takes most of the computation time !
        keys_it = A.begin()

        # Iterates over keys in left matrix
        while(keys_it != A.end()):
            tmp = 0.0
            # Iterates over smallest set of values between A and v
            if A.size() > dereference(keys_it).second.size():
                values_it = dereference(keys_it).second.begin()
                while(values_it != dereference(keys_it).second.end()):
                    # Computes dot-product
                    tmp += dereference(values_it).second * v_map[dereference(values_it).first]
                    postincrement(values_it)
            else:
                values_v_it = v_map.begin()
                values = dereference(keys_it).second
                while(values_v_it != v_map.end()):
                    # Computes dot-product
                    tmp += dereference(values_v_it).second * values[dereference(values_v_it).first]
                    postincrement(values_v_it)

            # Update result
            if tmp != 0.0:
                res[dereference(keys_it).first] = tmp
            postincrement(keys_it)

        return res

    def __str__(self):
        return f'{self.my_dict}'

    def __dict2map(self, dic: dict):
        cdef unordered_map[int, float] dic_map = dic
        return dic_map

    def __csr2map(self, adjacency: Union[sparse.csr_matrix, np.ndarray]):
        cdef unordered_map[int, unordered_map[int, int]] adj_map
        cdef unordered_map[int, int] sub_adj_map
        cdef int i
        cdef int len_indptr = len(adjacency.indptr)
        cdef vector[int] columns
        cdef vector[int] data

        for i in range(1, len_indptr):
            if (adjacency.indptr[i] - adjacency.indptr[i-1]>0):
                columns = adjacency.indices[adjacency.indptr[i-1]:adjacency.indptr[i]]
                data = adjacency.data[adjacency.indptr[i-1]:adjacency.indptr[i]]
                sub_adj_map.clear() # Clearing unordered map at each step
                for col, val in zip(columns, data):
                    sub_adj_map[col] = val
                adj_map[i-1] = sub_adj_map

        return adj_map
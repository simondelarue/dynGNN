## Speed tests for data structure choice

In order to chose the right data structure to use when dealing with streams of graphs over time, we performed some speed-tests for basic operations :
* Sparse Matrix - Dense Vector multiplication : `SpMDV`  
* Sparse Matrix - Sparse Vector multiplcation : `SpMSpV`

Several combinations of the following parameters have been tested :
* density of sparse matrix
* number of vertices in graph
* data structure of matrix and vector

We considered the following data structures :
* `CSR`, `CSC`, `COO`, `LIL`, `DOK` : sparse formats from `scipy` library
* `python_dict` : ajdacency matrix of the graph is stored as a python dictionary of dictionaries
* `cython_dict` : cythonized version of `python_dict`, where the underlying structure used is a C++ `unordered map`
* `cython_dict_map` : cythonized version of `python_dict`, where the underlying structure used is a C++ `map`

### Usage

Compile cython code then execute python script with the following command lines :

``` bash
python3 setup.py build_ext --inplace
python3 main.py
```

### Results

Results are available in `stream.ipynb` file.

''' Utils module '''

from stream.utils.parse import edgelist2adjacency
from stream.utils.format import bipartite2undirected
from stream.utils.neighbors import get_neighbors
from stream.utils.update import add_edges

class Bunch(dict):
    ''' Container object for datasets.
    Dictionary-like object that exposes its keys as attributes.

    This code is taken from scikit-learn.
    >>> bunch = Bunch(a=1, b=2)
    >>> bunch['a']
    1
    >>> bunch.a
    1
    >>> bunch.b = 3
    >>> bunch['b']
    3
    >>> bunch.c = 4
    >>> bunch['c']
    4
    '''

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
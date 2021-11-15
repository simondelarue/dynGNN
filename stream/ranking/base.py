#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Oct 20, 2021
@author: Simon Delarue <sdelarue@enst.fr>

This code is heavily inspired from scikit-network package.
'''

from abc import ABC

import numpy as np

class BaseRanking(ABC):
    def __init__(self):
        self.scores_ = None

    def fit_transform(self, *args, **kwargs) -> np.ndarray:
        ''' Fit algorithm to data and return the scores. Same parameters as the ``fit`` method.

        Returns
        -------
        scores : np.ndarray
            Scores.
        '''

        self.fit(*args, **kwargs)
        return self.scores_

    def update_transform(self, *args, **kwargs) -> np.ndarray:
        self.update(*args, **kwargs)
        #self.update_selected(*args, **kwargs)
        return self.scores_

    def _split_vars(self, shape):
        n_row = shape[0]
        self.scores_row_ = self.scores_[:n_row]
        self.scores_col_ = self.scores_[n_row:]
        #self.scores_ = self.scores_row_
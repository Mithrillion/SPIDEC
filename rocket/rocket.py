# -*- coding: utf-8 -*-
"""Rocket transformer."""
"""modified from https://github.com/sktime/sktime/blob/main/sktime/transformations/panel/rocket/_rocket.py"""

import multiprocessing

import numpy as np
import pandas as pd


class Rocket:
    def __init__(self, num_kernels=10_000, normalise=True, n_jobs=1, random_state=None):
        self.num_kernels = num_kernels
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state if isinstance(random_state, int) else None

    def fit(self, X, y=None):
        from ._rocket_numba import _generate_kernels

        _, self.n_columns, n_timepoints = X.shape
        self.kernels = _generate_kernels(
            n_timepoints, self.num_kernels, self.n_columns, self.random_state
        )
        return self

    def transform(self, X, y=None):
        from numba import get_num_threads, set_num_threads

        from ._rocket_numba import _apply_kernels

        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )
        prev_threads = get_num_threads()
        if self.n_jobs < 1 or self.n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()
        else:
            n_jobs = self.n_jobs
        set_num_threads(n_jobs)
        t = _apply_kernels(X.astype(np.float32), self.kernels)
        set_num_threads(prev_threads)
        return t
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

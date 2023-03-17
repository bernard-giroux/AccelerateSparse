#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:37:17 2023

@author: giroux
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pyAccelerate.sparse_solve import DirectSolver, is_symmetric




A = sp.csc_matrix(np.array([[10.0, 1.0, 0.0, 2.5],
                            [1.0, 12., -0.3, 1.1],
                            [0.0, -0.3, 9.5, 0.0],
                            [2.5,  1.1, 0.0, 6.0]]))
Al = sp.csc_matrix(np.array([[10.0, 0.0, 0.0, 0.0],
                            [1.0, 12.,  0.0, 0.0],
                            [0.0, -0.3, 9.5, 0.0],
                            [2.5,  1.1, 0.0, 6.0]]))

print(is_symmetric(A))

print(is_symmetric(Al))


# %%
dc = DirectSolver(verbose=True)

dc.factorize(A)

b = np.array([2.2, 2.85, 2.79, 2.87])

x = dc.solve(b)

x2 = spsolve(A, b)


print(x, A @ x - b)
print(x2, A @ x2 - b)

# %%

A[1, 0] = A[0, 1] = 2.0
dc.refactor(A)

x = dc.solve(b)

x2 = spsolve(A, b)


print(x, A @ x - b)
print(x2, A @ x2 - b)

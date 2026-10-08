# -*- coding: utf-8 -*-
"""
Week 2 Labs, Quantum Many Body

Created on Thu Oct  8 15:16:36 2026

@author: Maks Selevic, Joe Martin
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import scipy.sparse as sparse

# Function Definitions
def create_operators_sparse(size):
    sx_sparse = sparse.csr_matrix([[0,0.5],[0.5,0]], dtype=complex)
    sy_sparse = sparse.csr_matrix([[0, -0.5j],[0.5j,0]], dtype=complex)
    sz_sparse = sparse.csr_matrix([[0.5,0],[0,-0.5]], dtype=complex)
    
    identity = sparse.identity(2, format="csr", dtype=complex)

    sx_operators = []
    sy_operators = []
    sz_operators = []

    for j in range(size):
        factors_x = [identity]*size
        factors_y = [identity]*size
        factors_z = [identity]*size

        factors_x[j] = sx_sparse
        factors_y[j] = sy_sparse
        factors_z[j] = sz_sparse

        operator_x = factors_x[0]
        operator_y = factors_y[0]
        operator_z = factors_z[0]

        for factor_x in factors_x[1:]:
            operator_x = sparse.kron(operator_x, factor_x, format="csr")
        for factor_y in factors_y[1:]:
            operator_y = sparse.kron(operator_y, factor_y, format="csr")
        for factor_z in factors_z[1:]:
            operator_z = sparse.kron(operator_z, factor_z, format="csr")

        sx_operators.append(operator_x)
        sy_operators.append(operator_y)
        sz_operators.append(operator_z)

    return sx_operators, sy_operators, sz_operators
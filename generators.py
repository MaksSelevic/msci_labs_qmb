# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:47:01 2026

A file used to generate and save spin matrices for ease of use.

@author: Maks
"""
# Imports
import numpy as np
import scipy.sparse as sparse


def create_operators_dense(size):
    sx = 0.5*np.array([[0,1],[1,0]])
    sy = 0.5*np.array([[0, -1.0j],[1.0j,0]])
    sz = 0.5*np.array([[1,0],[0,-1]])
    identity = np.identity(2)

    sx_operators = []
    sy_operators = []
    sz_operators = []

    for j in range(size):
        factors_x = [identity]*size
        factors_y = [identity]*size
        factors_z = [identity]*size

        factors_x[j] = sx
        factors_y[j] = sy
        factors_z[j] = sz

        operator_x = factors_x[0]
        operator_y = factors_y[0]
        operator_z = factors_z[0]

        for factor_x in factors_x[1:]:
            operator_x = np.kron(operator_x, factor_x)
        for factor_y in factors_y[1:]:
            operator_y = np.kron(operator_y, factor_y)
        for factor_z in factors_z[1:]:
            operator_z = np.kron(operator_z, factor_z)

        sx_operators.append(operator_x)
        sy_operators.append(operator_y)
        sz_operators.append(operator_z)

    return sx_operators, sy_operators, sz_operators

def create_operators_sparse(size):
    sx = sparse.csr_matrix([[0,0.5],[0.5,0]], dtype=complex)
    sy = sparse.csr_matrix([[0, -0.5j],[0.5j,0]], dtype=complex)
    sz = sparse.csr_matrix([[0.5,0],[0,-0.5]], dtype=complex)
    
    identity = sparse.identity(2, format="csr", dtype=complex)

    sx_operators = []
    sy_operators = []
    sz_operators = []

    for j in range(size):
        factors_x = [identity]*size
        factors_y = [identity]*size
        factors_z = [identity]*size

        factors_x[j] = sx
        factors_y[j] = sy
        factors_z[j] = sz

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

def generate_hamiltonian(size):
    sx_ops, sy_ops, sz_ops = create_operators_sparse(size)
    
    dimension = 2**size   
    hamiltonian = sparse.csr_matrix((dimension, dimension), dtype = complex)

    for i in range(size):
        pair_hamiltonian = (sx_ops[i]@sx_ops[(i+1)%size]+
                            sy_ops[i]@sy_ops[(i+1)%size]+
                            sz_ops[i]@sz_ops[(i+1)%size])
        
        hamiltonian += pair_hamiltonian
    return hamiltonian

# Generating Hamiltonians

for i in range(1, 5+1):
    hamiltonian = generate_hamiltonian(i)
    sparse.save_npz(f"operators\h_{i}_sparse", hamiltonian)
    #np.save(f"operators\h_{i}_dense", hamiltonian.toarray())
    


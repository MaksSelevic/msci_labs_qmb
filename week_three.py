# -*- coding: utf-8 -*-
"""
Week 2 Labs, Quantum Many Body

Created on Thu Oct  8 15:16:36 2026

@author: Maks Selevic, Joe Martin
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
#from time import perf_counter
import scipy.sparse as sparse

# Function Definitions
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

def plot_table(matrix, title):
    fig, ax = plt.subplots()
    norm = plt.Normalize(-1, 1)
    table = ax.table(cellText=matrix,
                     loc=(0,0), cellLoc="center",
                     cellColours=plt.cm.coolwarm_r(norm(np.abs(matrix))))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            cell = table[i, j]
            cell.set_height(1/len(matrix))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

###############################################################################
# Problem 8
###############################################################################
#hamiltonian_test = generate_hamiltonian(2)
#plot_table(hamiltonian_test.toarray().real, "Hamiltonian of a two particle system")

###############################################################################
# Problem 9
###############################################################################

hamiltonian = generate_hamiltonian(6)

eigvals_sparse, eigvecs_sparse = sparse.linalg.eigsh(hamiltonian, k=16, which="SA")
order_sparse = np.argsort(eigvals_sparse)
eigvals_sparse = eigvals_sparse[order_sparse]
eigvecs_sparse = eigvecs_sparse[:, order_sparse]

eigvals_dense, eigvecs_dense = np.linalg.eigh(hamiltonian.toarray())
order_dense = np.argsort(eigvals_dense)
eigvals_dense = eigvals_dense[order_dense]
eigvecs_dense = eigvecs_dense[:, order_dense]

eigval_residuals = eigvals_dense[:len(eigvals_sparse)] - eigvals_sparse

print("Are the eigenvalues equivalent for both methods?")
for i in range(len(eigvals_sparse)):
    close_check = np.isclose(eigvals_sparse[i], eigvals_dense[i])
    print(f"Eigenvalue {i+1}: {"Yes!" if close_check else "No!"}")
    
plt.scatter(range(1, len(eigval_residuals)+1), eigval_residuals)
plt.xlabel(f"First {len(eigvals_sparse)} eigenvalues of the Hamiltonian")
plt.ylabel("Residuals")
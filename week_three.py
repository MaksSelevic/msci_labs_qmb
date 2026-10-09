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

def eigen_extractor(matrix, matrix_type="sparse", n_eigvals=20):

    if matrix_type == "sparse":
        eigvals, eigvecs = sparse.linalg.eigsh(matrix, k=n_eigvals, which="SA")
    elif matrix_type == "dense":
        eigvals, eigvecs = np.linalg.eigh(matrix)
    else:
        print("Incorrect matrix type chosen, use sparse or dense.")
        return np.zeros(n_eigvals), np.zeros((n_eigvals, n_eigvals))

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    return eigvals, eigvecs


###############################################################################
# Problem 8
###############################################################################
hamiltonian_test = sparse.load_npz("operators\h_2_sparse.npz")
plot_table(hamiltonian_test.toarray().real, "Hamiltonian of a two particle system")

###############################################################################
# Problem 9
###############################################################################

hamiltonian_10_sparse = sparse.load_npz("operators\h_10_sparse.npz")
hamiltonian_10_dense = np.load("operators\h_10_dense.npy")
hamiltonian_11_sparse = sparse.load_npz("operators\h_11_sparse.npz")
hamiltonian_11_dense = np.load("operators\h_11_dense.npy")

test_hamiltonian_sparse = hamiltonian_11_sparse
test_hamiltonian_dense = hamiltonian_11_dense

eigvals_sparse, eigvecs_sparse = eigen_extractor(test_hamiltonian_sparse)
eigvals_dense, eigvecs_dense = eigen_extractor(test_hamiltonian_dense, "dense")

print("Are the eigenvalues equivalent for both methods?")
for i in range(len(eigvals_sparse)):
    close_check = np.isclose(eigvals_sparse[i], eigvals_dense[i])
    print(f"Eigenvalue {i+1}: {"Yes!" if close_check else "No!"}")


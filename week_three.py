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
        eigvals, eigvecs = sparse.linalg.eigsh(matrix, k=n_eigvals, ncv=10*n_eigvals, which="SA")
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        return eigvals, eigvecs
    elif matrix_type == "dense":
        eigvals, eigvecs = np.linalg.eigh(matrix)
        return eigvals, eigvecs
    else:
        print("Incorrect matrix type chosen, use sparse or dense.")
        return np.zeros(n_eigvals), np.zeros((n_eigvals, n_eigvals))


###############################################################################
# Problem 8
###############################################################################
#hamiltonian_test = sparse.load_npz("operators\h_2_sparse.npz")
#plot_table(hamiltonian_test.toarray().real, "Hamiltonian of a two particle system")

###############################################################################
# Problem 9
###############################################################################

#hamiltonian_10_sparse = sparse.load_npz("operators\h_10_sparse.npz")
#hamiltonian_10_dense = np.load("operators\h_10_dense.npy")
#hamiltonian_11_sparse = sparse.load_npz("operators\h_11_sparse.npz")
#hamiltonian_11_dense = np.load("operators\h_11_dense.npy")

#test_hamiltonian_sparse = hamiltonian_11_sparse
#test_hamiltonian_dense = hamiltonian_11_dense

#eigvals_sparse_test, eigvecs_sparse_test = eigen_extractor(test_hamiltonian_sparse)
#eigvals_dense_test, eigvecs_dense_test = eigen_extractor(test_hamiltonian_dense, "dense")

#print("Are the eigenvalues equivalent for both methods?")
#for i in range(len(eigvals_sparse_test)):
#    close_check = np.isclose(eigvals_sparse_test[i], eigvals_dense_test[i])
#    print(f"Eigenvalue {i+1}: {"Yes!" if close_check else "No!"}")

###############################################################################

min_n, max_n = 4, 12
n_eigvals = 12

residuals = []

for i in range(min_n, max_n+1):
    hamiltonian_sparse = sparse.load_npz(f"operators\h_{i}_sparse.npz")
    hamiltonian_dense = np.load(f"operators\h_{i}_dense.npy")

    eigvals_sparse, eigvecs_sparse = eigen_extractor(hamiltonian_sparse,
                                                     "sparse",
                                                     n_eigvals=n_eigvals)
    eigvals_dense, eigvecs_dense = eigen_extractor(hamiltonian_dense,
                                                   "dense",
                                                   n_eigvals=n_eigvals)
    eigvals_residuals = eigvals_dense[:n_eigvals] - eigvals_sparse
    residuals.append(eigvals_residuals)

for i in range(max_n-min_n+1):
    plt.scatter(range(1, n_eigvals+1), residuals[i], label=f"{i+min_n}")
plt.xlabel(f"First {n_eigvals} eigenvalues")
plt.ylabel("Residuals of the eigenvalues")
plt.legend(title="Number of Particles:", ncol=3)
plt.show()

residuals_averages = []

for i in range(max_n-min_n+1):
    res_average = np.sum(abs(residuals[i]))/len(residuals[i])
    residuals_averages.append(res_average)

plt.scatter(range(min_n, max_n+1), residuals_averages, color="b")
plt.xlabel("Number of Particles")
plt.ylabel("Average Absolute Value of Residuals")
plt.show()

m, c = np.polyfit(np.log(range(min_n, max_n+1)), np.log(residuals_averages), 1)

plt.scatter(np.log(range(min_n, max_n+1)), np.log(residuals_averages), color="b")
plt.plot(np.log(range(min_n, max_n+1)), m*(np.log(range(min_n, max_n+1)))+c, color="r", label=(f"Slope: {m:.2f}"+"\n"+f"Intercept: {c:.2f}"))
plt.xlabel("log(Number of Particles)")
plt.ylabel("log(Average Absolute Value of Residuals)")
plt.legend()
plt.show()

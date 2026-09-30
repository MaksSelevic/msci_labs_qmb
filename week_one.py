# -*- coding: utf-8 -*-
"""
Week 1 labs, Quantum Many Body

Created on Thu Sep 24 12:16:27 2026

@author: Maks Selevic, Joe Martin
"""
# Imports
import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt

# Function definitions
def check_hermitian(matrix):
    state = lin.ishermitian(matrix)
    if state == True:
        result = "Yes!"
    elif state == False:
        result = "No!"
    print(f"Is the calculated Hamiltonian Hermitian? {result}"+"\n")

def plot_table(matrix, title):
    fig, ax = plt.subplots()
    norm = plt.Normalize(-1, 1)
    table = ax.table(cellText=matrix,
                     loc=(0,0), cellLoc="center",
                     cellColours=plt.cm.coolwarm_r(norm(matrix)))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            cell = table[i, j]
            cell.set_height(1/len(matrix))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

# Variables
# Spin elements in x, y and z directions
sx = 0.5*np.array([[0, 1], [1,0]])
sy = 0.5*np.array([[0, -1.0j], [1.0j, 0]])
sz = 0.5*np.array([[1, 0], [0, -1]])

###############################################################################
# Problem 2
###############################################################################

# Tensor products of the spin elemets with themselves
sx_sx = np.kron(sx, sx)
sy_sy = np.kron(sy, sy)
sz_sz = np.kron(sz, sz)

# Calculating the Hamiltonian using equation 14 (J=1)
hamiltonian = np.sum((sx_sx,sy_sy, sz_sz), axis=0)

# Checking for hermiticity
check_hermitian(hamiltonian)

# Creating a table showing the Hamiltonian
plot_table(hamiltonian.real, "The Hamiltonian")

###############################################################################
# Problem 3
###############################################################################

# Extracting the eigenvalues and the eigenvectors:
eigvals, eigvecs = np.linalg.eigh(hamiltonian)

print(f"The eigenvalues of the Hamiltonian are {eigvals}")

plot_table(np.round(eigvecs.real, 4), "Eigenvectors of the Hamiltonian")

###############################################################################
# Problem 4
###############################################################################





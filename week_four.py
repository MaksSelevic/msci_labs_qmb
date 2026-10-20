# -*- coding: utf-8 -*-
"""
Week 4 Labs, Quantum Many Body

Created on Thu Oct 15 10:24:12 2026

@author: Maks Selevic, Joe Martin
"""

# Imports
from math import comb, ceil
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

# Functions
def plot_table(matrix, title):
    fig, ax = plt.subplots()
    norm = plt.Normalize(-1, 1)
    if np.all(matrix == np.conj(matrix)):
        matrix = matrix.real
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

#def sector_states(L, n_up):
#        
#    all_states = np.arange(2**L)
#    number_of_ones = np.zeros(2**L, dtype = int)
#    for i in range(L):
#        number_of_ones += (all_states >> i) & 1
#
#    return all_states[number_of_ones == n_up]

def sector_states(n_particles, n_up):
    return [index for index in range(2**n_particles) if index.bit_count()==n_up]

def order(L):
    index_order = np.empty(0, dtype=int)

    for i in range(L+1):
        indeces = np.array(sector_states(L, i))
        index_order = np.concatenate((index_order, indeces))
    return index_order

def block_diagonalise_dense(matrix, n_particles):
    index_order = order(n_particles)
    size = 2**n_particles
    h_row = np.zeros((size,size), dtype=complex)
    h_ord = np.zeros((size,size), dtype=complex)

    for i, j in enumerate(index_order):
        h_row[i] = matrix[j]

    for i, j in enumerate(index_order):
        h_ord[:, i] = h_row[:, j]

    return h_ord

def block_diagonalise_sparse(matrix, n_particles):
    index_order = order(n_particles)
    size = 2**n_particles
    h_row = sparse.csr_matrix((size, size), dtype = complex)
    h_ord = sparse.csr_matrix((size, size), dtype = complex)

    for i, j in enumerate(index_order):
        h_row[i] = matrix[j]

    for i, j in enumerate(index_order):
        h_ord[:, i] = h_row[:, j]
    
    return h_ord

def create_operators(size):
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

# Variables
#dense_3 = np.load("operators\h_3_dense.npy")
#dense_4 = np.load("operators\h_4_dense.npy")
#dense_5 = np.load("operators\h_5_dense.npy")
#dense_6 = np.load("operators\h_6_dense.npy")

#sparse_6 = sparse.load_npz("operators\h_6_sparse.npz")

#ordered_3 = block_diagonalise_dense(dense_3, 3)
#plot_table(dense_3, "3 Particles")
#plot_table(ordered_3, "3 Particles (ordered)")

#ordered_4 = block_diagonalise_dense(dense_4, 4)
#plot_table(dense_4, "4 Particles")
#plot_table(ordered_4, "4 Particles (ordered)")

#ordered_5 = block_diagonalise_dense(dense_5, 5)
#plot_table(dense_5, "5 Particles")
#plot_table(ordered_5, "5 Particles (ordered)")

#ordered_6 = block_diagonalise_dense(dense_6, 6)
#plot_table(dense_6, "6 Particles")
#plot_table(ordered_6, "6 Particles (ordered)")

#sp_ordered_6 = block_diagonalise_sparse(sparse_6, 6)
#plot_table(sparse_6.toarray(), "6 Particles (sp)")
#plot_table(sp_ordered_6.toarray(), "6 Particles (ordered)(sp)")


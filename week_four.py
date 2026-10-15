# -*- coding: utf-8 -*-
"""
Week 4 Labs, Quantum Many Body

Created on Thu Oct 15 10:24:12 2026

@author: Maks Selevic, Joe Martin
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Functions
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
dense_2 = np.load("operators\h_2_dense.npy")
dense_3 = np.load("operators\h_3_dense.npy")
dense_4 = np.load("operators\h_4_dense.npy")

plot_table(dense_2.real, "2 Particles")
plot_table(dense_3.real, "3 Particles")
plot_table(dense_4.real, "4 Particles")

sx, sy, sz = create_operators(3)
sx12 = np.dot(sx[0], sx[1])
sx23 = np.dot(sx[1], sx[2])
sx31 = np.dot(sx[2], sx[0])

sy12 = np.dot(sy[0], sy[1])
sy23 = np.dot(sy[1], sy[2])
sy31 = np.dot(sy[2], sy[0])

sz12 = np.dot(sz[0], sz[1])
sz23 = np.dot(sz[1], sz[2])
sz31 = np.dot(sz[2], sz[0])

H = sx12+sx23+sx31+sy12+sy23+sy31+sz12+sz23+sz31

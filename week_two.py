# -*- coding: utf-8 -*-
"""
Week 2 labs, Quantum Many Body

Created on Thu Sep 24 12:16:27 2026

@author: Maks Selevic, Joe Martin
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Function Definitions
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

def evaluate_commutator(A, B, expectation):
    commutator = np.dot(A,B)-np.dot(B,A)
    return np.allclose(commutator, expectation)


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
sx_operators, sy_operators, sz_operators = create_operators(4)

# Main Code
#plot_table(sx_operators[2], "Sx operator of the third particle")

sx2_sz2 = evaluate_commutator(sx_operators[2], sz_operators[2], -1j*sy_operators[2])
print(f"Is the Sx2 and Sz2 commutator equivalent to -i*Sy2? {"Yes!" if sx2_sz2 else "No!"}")


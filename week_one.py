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
    print(f"Is the calculated Hamiltonian Hermitian? {"Yes!" if state else "No!"}"+"\n")

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

def check_normalisation(state):
    norm = np.linalg.norm(state)
    return np.isclose(norm, 1)

def randomised_state(n):
    coefficients = np.zeros(n, dtype = complex)
    for n in range(n):
        r = np.random.uniform(0, 1)
        theta = np.random.uniform(0, 2*np.pi)
        coefficients[n] = r*np.cos(theta)+1.0j*r*np.sin(theta)
    coefficients = coefficients/np.linalg.norm(coefficients)
    return coefficients

def rayleigh_quotient(state, hamiltonian):
    bra = state.conj().T
    coefficient = np.dot(np.dot(bra,hamiltonian), state)
    return coefficient.real

def add_perturbation():
    pass

# Variables
# Spin elements in x, y and z directions
sx = 0.5*np.array([[0, 1], [1,0]])
sy = 0.5*np.array([[0, -1.0j], [1.0j, 0]])
sz = 0.5*np.array([[1, 0], [0, -1]])

# Trial States
trial_singlet = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
trial_triplet_uu = np.array([1, 0, 0, 0])
trial_product = randomised_state(4)

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

# Plotting the table of eigenvectors for easier visualisation
plot_table(np.round(eigvecs.real, 4), "Eigenvectors of the Hamiltonian")

###############################################################################
# Problem 4
###############################################################################
print(f"""
Checking if the trial states are normalised:
Singlet (ground) state: {'Yes' if check_normalisation(trial_singlet) == True else "No"}
Pure up/up state: {'Yes' if check_normalisation(trial_triplet_uu) == True else "No"}
Randomised product state: {'Yes' if check_normalisation(trial_product) == True else "No"}
""")

print(f"""The Rayleigh quotient of the trial states are:
{rayleigh_quotient(trial_singlet, hamiltonian): .2f} for the singlet (ground) state,
{rayleigh_quotient(trial_triplet_uu, hamiltonian): .2f} for the pure up/up state,
{rayleigh_quotient(trial_product, hamiltonian): .2f} for the randomised product state.
""")



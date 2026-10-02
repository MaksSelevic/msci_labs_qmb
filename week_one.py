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

def perturbation(state, perturbation_weight):
    perturbation = randomised_state(4)
    perturbation -= np.dot(state, perturbation)*state
    perturbation /= np.linalg.norm(perturbation)
    #perturbed_state = np.sqrt(1-perturbation_weight)*state+np.sqrt(perturbation_weight)*perturbation
    perturbed_state = np.cos(perturbation_weight)*state+np.sin(perturbation_weight)*perturbation
    return perturbed_state/np.linalg.norm(perturbed_state)

def progressive_perturbation(n, state, hamiltonian, max_perturbation):
    energy_differences = []
    perturbation_size = []
    state_energy = rayleigh_quotient(state, hamiltonian)
    for perturbation_weight in np.linspace(10e-4, max_perturbation, n):
        perturbed_state = perturbation(state, perturbation_weight)#, perturbation_direction)
        energy_difference = rayleigh_quotient(perturbed_state, hamiltonian) - state_energy
        energy_differences.append(energy_difference)
        perturbation_size.append(perturbation_weight)
    return np.array(energy_differences), np.array(perturbation_size)    

# Variables
# Spin elements in x, y and z directions
sx = 0.5*np.array([[0, 1], [1,0]])
sy = 0.5*np.array([[0, -1.0j], [1.0j, 0]])
sz = 0.5*np.array([[1, 0], [0, -1]])

# Trial States
trial_singlet = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
trial_triplet_uu = np.array([1, 0, 0, 0])
trial_product = randomised_state(4)

trial_perturbation_direction = trial_triplet_uu
trial_perturbation_weight = 0.1
trial_perturbed_state = perturbation(trial_singlet, trial_perturbation_weight)#, trial_perturbation_direction)

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
Perturbed state ({(1.0-trial_perturbation_weight)*100:.0f}% Ground State): {'Yes' if check_normalisation(trial_perturbed_state) == True else "No"}
""")

print(f"""The Rayleigh quotient of the trial states are:
{rayleigh_quotient(trial_singlet, hamiltonian): .2f} for the singlet (ground) state,
{rayleigh_quotient(trial_triplet_uu, hamiltonian): .2f} for the pure up/up state,
{rayleigh_quotient(trial_product, hamiltonian): .2f} for the randomised product state.
{rayleigh_quotient(trial_perturbed_state, hamiltonian): .2f} for the trial perturbed state. ({(1.0-trial_perturbation_weight)*100:.0f}% Ground State)
""")

y, x = progressive_perturbation(100, trial_singlet, hamiltonian, np.pi/50)

plt.show()
a, s, d = np.polyfit(x, y, 2)
plt.scatter(x, y)
plt.plot(x, a*x*x+s*x+d, color="red", label=f"Quadratic coefficients: ({a:.2f})x^2+({s:.2f})x+({d:.2f})")
plt.title("Perturbation Amplitude vs Energy Difference")
plt.legend()
plt.show()

plt.scatter(np.log(x),np.log(y))
plt.title("Log of Perturbation Amplitude vs Log of Energy Difference")
m, c = np.polyfit(np.log(x), np.log(y), 1)
plt.plot(np.log(x), m*np.log(x)+c, color="red", label=f"Gradient of {m:.3f}")
plt.legend()
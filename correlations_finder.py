# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 09:50:29 2026

@author: Maks-
"""

# Imports
import numpy as np
from math import ceil
from scipy import sparse
import matplotlib as mpl
import matplotlib.pyplot as plt

# Functions
def create_operator_sparse(size):
    sz = sparse.csr_matrix([[0.5,0],[0,-0.5]], dtype=complex)
    identity = sparse.identity(2, format="csr", dtype=complex)
    sz_operators = []

    for j in range(size):
        factors_z = [identity]*size
        factors_z[j] = sz
        operator_z = factors_z[0]

        for factor_z in factors_z[1:]:
            operator_z = sparse.kron(operator_z, factor_z, format="csr")

        sz_operators.append(operator_z)

    return sz_operators

def sector_states(n_particles, n_up):
    return [index for index in range(2**n_particles) if index.bit_count()==n_up]


def projection_sparse(matrix, n_particles):
    indices = sector_states(n_particles, ceil(n_particles/2))
    projection = sparse.lil_matrix((2**n_particles, len(indices)), dtype=float)

    for i, j in enumerate(indices):
        projection[j, i] = 1.0

    return projection.transpose() @ matrix @ projection

def sector_states_fast(n_particles, n_up):
    # Same as sector_states but using numpy arrays instead of a python loop
    all_states = np.arange(2**n_particles, dtype=np.int64)
    return all_states[np.bitwise_count(all_states) == n_up]

def zz_correlation(state, L, n_up):
    # C(r) = <S^z_0 S^z_r>, averaged over every starting site j since the ring is
    # translation invariant (less rounding noise than using site 0 only)
    basis = sector_states_fast(L, n_up)
    probability = np.abs(state)**2
    spins = np.array([((basis >> j) & 1) - 0.5 for j in range(L)])   # S^z_j for every basis state
    correlation = np.zeros(L//2 + 1)
    for r in range(L//2 + 1):
        for j in range(L):
            correlation[r] += np.sum(probability * spins[j] * spins[(j+r) % L])
    return correlation/L

# Main Code
# Variables
correlations = {}
min_n = 4
max_n = 26
step_size = 2

# Generating the correlations
#for i in range(min_n, max_n+1, step_size):
#    state = np.load(f"results\ground_state_{i}.npy").T
#    correlations = zz_correlation(state, i, i//2)
#    np.save(f"results\correlations_{i}", correlations)

# Plotting the correlations
for i in range(min_n, max_n+1, step_size):
    correlations[i] = np.load(f"results\correlations_{i}.npy")

cmap = mpl.colormaps["rainbow"]
colors = cmap(np.linspace(0, 1, len(correlations)))

fig, ax = plt.subplots()
for key in correlations:
    ax.plot(np.arange(key//2 + 1), correlations[key], marker="o", ls="-", ms=5, color = colors[(key//step_size)-(min_n//step_size)], label=f"n = {key}")
ax.plot(np.arange(max_n//2 + 1), 0.25*(-1.0)**np.arange(max_n//2 + 1), "k:", label="Neel")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("r")
ax.set_ylabel(r"$C(r) = \langle S^z_0 S^z_r \rangle$")
plt.legend(fontsize=8, ncol=2)
plt.show()


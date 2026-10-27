# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:45:56 2026

Script used to extract the first two energy levels of n-particle systems.

@author: Maks
"""

# Imports
from scipy import sparse
import numpy as np

# Main Code

n_particles_max = 26
e_even = np.zeros((2, int((n_particles_max-2)/2)))
indices_even = np.arange(4, n_particles_max+1, 2)

for i in indices_even:
    hamiltonian = sparse.load_npz(f"operators\h_{i}_mid_sparse.npz")
    e = sparse.linalg.eigsh(hamiltonian, k=2, ncv=20, which="SA", return_eigenvectors=False)
    e_even[:,int((i-3)/2)] = e
e_indexed = np.stack((indices_even, e_even[0], e_even[1]))
np.save(r"results\first_two_e_even.npy", e_indexed)
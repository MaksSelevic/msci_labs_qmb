# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:13:08 2026

Script used for extracting the ground states of n-particle systems and finding
their energy density.

@author: Maks Selevic, Joe Martin
"""
# Imports
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt

# Main code
n_particles_max = 26
e_density_even = np.zeros(int((n_particles_max-2)/2))
e_density_odd = np.zeros(int((n_particles_max-2)/2))

for i in range(3, n_particles_max, 2):
    hamiltonian = sparse.load_npz(f"operators\h_{i}_mid_sparse.npz")
    e_ground = sparse.linalg.eigsh(hamiltonian, k=1, ncv=10, which="SA", return_eigenvectors=False)
    e_density_odd[int((i-3)/2)] = e_ground[0]/i
#np.save("results\e_density_ground_odd.npy", e_density_odd)

for i in range(4, n_particles_max+1, 2):
    hamiltonian = sparse.load_npz(f"operators\h_{i}_mid_sparse.npz")
    e_ground = sparse.linalg.eigsh(hamiltonian, k=1, ncv=10, which="SA", return_eigenvectors=False)
    e_density_even[int((i-3)/2)] = e_ground[0]/i
#np.save("results\e_density_ground_even.npy", e_density_even)

plt.scatter(range(3, n_particles_max, 2),
            e_density_odd,
            s=20, color="darkorange",
            label="Odd n")
plt.scatter(range(4, n_particles_max+1, 2),
            e_density_even,
            s=20,
            color="r",
            label="Even n")
plt.plot(np.linspace(3, n_particles_max+0.5, 100),
         np.full(100, 0.25-np.log(2)),
         color="black",
         label="Analytical expectation",
         linestyle="--",
         linewidth=1.5,
         zorder=-1)
plt.xlabel("Number of partcles")
plt.ylabel("Energy density of the ground state")
plt.legend()
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:23:42 2026

Script used to visualise the ground state energy density.

@author: Maks Selevic, Joe Martin
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Functions
#def plot_graph(x, y, xlabel, ylabel):
#    fig, ax = plt.subplots()
#    ax.scatter(x, y, color="darkorange")
#    ax.set_xlabel(xlabel)
#    ax.set_ylabel(ylabel)

# Main code

e_odd = np.load("results\e_density_ground_odd.npy")
e_even = np.load("results\e_density_ground_even.npy")

n_odd = e_odd[0]
n_even = e_even[0]

n_inverse_odd = 1.0/n_odd
n_inverse_even = 1.0/n_even

n_inverse_square_odd = 1.0/(n_odd)**2
n_inverse_square_even = 1.0/(n_even)**2

plt.scatter(n_odd, e_odd[1], s=20, color="darkorange", label="Odd n")
plt.xlabel("Number of Particles")
plt.ylabel("Ground State Energy Density")
plt.legend()
plt.show()

plt.scatter(n_even, e_even[1], s = 20, color="r", label="Even n")
plt.xlabel("Number of Particles")
plt.ylabel("Ground State Energy Density")
plt.legend()
plt.show()

plt.scatter(n_inverse_odd, e_odd[1], s = 20, color="darkorange", label="Odd n")
plt.xlabel(r"$1/n$")
plt.ylabel("Ground State Energy Density")
plt.legend()
plt.show()

plt.scatter(n_inverse_even, e_even[1], s = 20, color="r", label="Even n")
plt.xlabel(r"$1/n$")
plt.ylabel("Ground State Energy Density")
plt.legend()
plt.show()

plt.scatter(n_inverse_square_odd, e_odd[1], s = 20, color="darkorange", label="Odd n")
plt.plot(np.linspace(0.0, n_inverse_square_odd[0], 100),
         np.full(100, 0.25-np.log(2)),
         color = "black",
         linestyle="--",
         label="Analytical Expectation")
plt.xlabel(r"$1/n^2$")
plt.ylabel("Ground State Energy Density")
plt.legend()
plt.show()

plt.scatter(n_inverse_square_even, e_even[1], s = 20, color="r", label="Even n")
plt.plot(np.linspace(0.0, n_inverse_square_even[0], 100),
         np.full(100, 0.25-np.log(2)),
         color = "black",
         linestyle="--",
         label="Analytical Expectation")
plt.xlabel(r"$1/n^2$")
plt.ylabel("Ground State Energy Density")
plt.legend()
plt.show()

coefs_odd, residuals_odd, rank_odd, singular_vals_odd, rcond_odd = np.polyfit(n_inverse_square_odd, e_odd[1], 1, full=True)
coefs_even, residuals_even, rank_even, singular_vals_even, rcond_even = np.polyfit(n_inverse_square_even, e_even[1], 1, full=True)

plt.scatter(n_inverse_square_odd, e_odd[1], s = 20, color="darkorange", label="Odd n")
plt.plot(n_inverse_square_odd, np.polyval(coefs_odd, n_inverse_square_odd), label="Polyfit")
plt.xlabel(r"$1/n^2$")
plt.ylabel("Ground State Energy Density")
plt.legend()
plt.show()

plt.scatter(n_inverse_square_even, e_even[1], s = 20, color="r", label="Even n")
plt.plot(n_inverse_square_even, np.polyval(coefs_even, n_inverse_square_even), label="Polyfit")
plt.xlabel(r"$1/n^2$")
plt.ylabel("Ground State Energy Density")
plt.legend()
plt.show()

###############################################################################
# Energy Difference
###############################################################################

e_first_two_even = np.load(r"results\first_two_e_even.npy")
n_inverse_diff = 1.0/e_first_two_even[0,1:]
e_diff = e_first_two_even[2,1:]-e_first_two_even[1,1:]
coefs_e_diff = np.polyfit(n_inverse_diff, e_diff, 1)

plt.scatter(n_inverse_diff,
            e_diff,
            s = 20,
            color="crimson",
            label=r"$\Delta E$")
plt.plot(n_inverse_diff,
         np.polyval(coefs_e_diff, n_inverse_diff),
         color="black",
         linestyle="--",
         label=f"""Slope: {coefs_e_diff[0]:0.3f}
Intercept: {coefs_e_diff[1]:0.3f}
""")
plt.xlabel(r"$1/n$")
plt.ylabel(r"$\Delta E$ between $|e\rangle$ and $|g\rangle$")
plt.legend()
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:47:07 2026

@author: Maks
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Plotting energy difference
e_first_two_even = np.load(r"results\first_two_e_even.npy")
n = e_first_two_even[0]
n_inverse = 1.0/n
e_diff = e_first_two_even[2]-e_first_two_even[1]
coefs_e_diff = np.polyfit(n_inverse, e_diff, 1)

plt.scatter(n_inverse,
            e_diff,
            s = 20,
            color="crimson",
            label=r"$\Delta E$")
plt.plot(n_inverse,
         np.polyval(coefs_e_diff, n_inverse),
         color="black",
         linestyle="--",
         label=f"""Slope: {coefs_e_diff[0]:0.3f}
Intercept: {coefs_e_diff[1]:0.3f}
""")
plt.xlabel(r"$1/n$")
plt.ylabel(r"$\Delta E$ between $|e\rangle$ and $|g\rangle$")
plt.legend()
plt.show()

# Finding y-intercepts
intercepts = np.zeros(len(n)-1)

for i in range(len(n)-1):
    slope, intercept = np.polyfit(n_inverse[i:], e_diff[i:], 1)
    intercepts[i] = intercept

fig, ax = plt.subplots()
ax2 = ax.twiny()
ax.scatter(n[:-1],
            intercepts,
            s = 20,
            color = "darkcyan",
            label = "Intercepts")
ax.plot(n[:-1],
        np.zeros(len(n)-1),
        color = "black",
        linestyle = "--",
        label = "Analytical expectation")
ax.set_xticks(range(4, 26, 2))
ax2.set_xlim(12.5, 1.5)
ax2.set_xticks(range(12, 1, -1))
ax2.set_xlabel("Number of data points used for the fit")
ax.set_xlabel("Number of particles of the smallest included system")
ax.set_ylabel(r"Intercept of the line of best fit of $\Delta E$ vs $1/n$")
ax.legend()
plt.show()
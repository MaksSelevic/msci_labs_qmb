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
def plot_graph(x, y, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="darkorange")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

# Main code

e_odd = np.load("results\e_density_ground_odd.npy")
e_even = np.load("results\e_density_ground_even.npy")


plot_graph(e_odd[0], e_odd[1], "Number of Particles", "Ground State Energy Density")
plot_graph(e_even[0], e_even[1], "Number of Particles", "Ground State Energy Density")
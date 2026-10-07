# -*- coding: utf-8 -*-
"""
Week 2 labs, Quantum Many Body

Created on Thu Sep 24 12:16:27 2026

@author: Maks Selevic, Joe Martin
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

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
                     cellColours=plt.cm.coolwarm_r(norm(np.abs(matrix))))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            cell = table[i, j]
            cell.set_height(1/len(matrix))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def operators_memory_usage_dense(n_particles, output_unit):
    n_elements = (2**n_particles)**2
    # Number of elements * bytes per element * 3 dimensions (xyz) * Number of particles
    memory_bytes = n_elements*16*3*n_particles 
    if output_unit == "b":
        return memory_bytes/8
    elif output_unit == "B":
        return memory_bytes
    elif output_unit == "KB":
        return memory_bytes/1024
    elif output_unit == "MB":
        return memory_bytes/(1024**2)
    elif output_unit == "GB":
        return memory_bytes/(1024**3)
    elif output_unit == "TB":
        return memory_bytes/(1024**4)
    else:
        return "Incorrect unit chosen. Use b, B, KB, MB, GB or TB."

def construction_time_trial(particle_range, repeats):
    trial_times = []
    for i in range(repeats):
        print(f"Construction time trial - loop {i+1} of {repeats}")
        times = np.zeros(len(particle_range))
        for n in particle_range:
            start = perf_counter()
            create_operators(n)
            end = perf_counter()
            times[n-2] = end-start
        trial_times.append(times)
    average_times = np.sum(trial_times, axis=0)/repeats
    return average_times, np.array(trial_times)

# Variables
n = 4
#sx_operators, sy_operators, sz_operators = create_operators(n)

memory_n_gb = 16
memory_n_mb = 11
memory_n_kb = 6

range_gb = range(10, memory_n_gb+1)
range_mb = range(5, memory_n_mb+1)
range_kb = range(1, memory_n_kb+1)

memory_needed_gb = []
memory_needed_mb = []
memory_needed_kb = []

n_construction_trial = 11
construction_trial_repeats = 4
trial_particles = range(2, n_construction_trial+1)
###############################################################################
# Problem 5
###############################################################################

#plot_table(sy_operators[3], f"Sy operator of the fourth particle, n = {n}")
#plot_table(sx_operators[2], f"Sx operator of the third particle, n = {n}")

#sx2_sz2 = evaluate_commutator(sx_operators[2], sz_operators[2], -1j*sy_operators[2])
#print(f"Is the Sx2 and Sz2 commutator equivalent to -i*Sy2? {"Yes!" if sx2_sz2 else "No!"}")

###############################################################################
# Problem 6
###############################################################################

#print(f"{operators_memory_usage_dense(13, "GB"):.3f}")

for n in range_gb:
    memory = operators_memory_usage_dense(n, "GB")
    memory_needed_gb.append(memory)

for n in range_mb:
    memory = operators_memory_usage_dense(n, "MB")
    memory_needed_mb.append(memory)

for n in range_kb:
    memory = operators_memory_usage_dense(n, "KB")
    memory_needed_kb.append(memory)

plt.scatter(range_gb, memory_needed_gb)
plt.xlabel("Number of particles")
plt.ylabel("Memory required for Operators (GB)")
plt.show()

plt.scatter(range_mb, memory_needed_mb)
plt.xlabel("Number of particles")
plt.ylabel("Memory required for Operators (MB)")
plt.show()

plt.scatter(range_kb, memory_needed_kb)
plt.xlabel("Number of particles")
plt.ylabel("Memory required for Operators (KB)")
plt.show()
    
#average_trial_times, trial_times = construction_time_trial(trial_particles, construction_trial_repeats)

#plt.scatter(trial_particles, average_trial_times)
#plt.xlabel("Number of Particles")
#plt.ylabel("Mean Construction Time of Operators (s)")
#plt.show()

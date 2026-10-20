# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 20:47:01 2026

A file used to generate and save spin matrices for ease of use.

@author: Maks
"""
# Imports
from time import perf_counter
from math import ceil, floor
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

def create_operators_dense(size):
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

def create_operators_sparse(size):
    sx = sparse.csr_matrix([[0,0.5],[0.5,0]], dtype=complex)
    sy = sparse.csr_matrix([[0, -0.5j],[0.5j,0]], dtype=complex)
    sz = sparse.csr_matrix([[0.5,0],[0,-0.5]], dtype=complex)
    
    identity = sparse.identity(2, format="csr", dtype=complex)

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
            operator_x = sparse.kron(operator_x, factor_x, format="csr")
        for factor_y in factors_y[1:]:
            operator_y = sparse.kron(operator_y, factor_y, format="csr")
        for factor_z in factors_z[1:]:
            operator_z = sparse.kron(operator_z, factor_z, format="csr")

        sx_operators.append(operator_x)
        sy_operators.append(operator_y)
        sz_operators.append(operator_z)

    return sx_operators, sy_operators, sz_operators

def generate_hamiltonian_sparse(size):
    sx_ops, sy_ops, sz_ops = create_operators_sparse(size)
    
    dimension = 2**size   
    hamiltonian = sparse.csr_matrix((dimension, dimension), dtype = complex)

    for i in range(size):
        pair_hamiltonian = (sx_ops[i]@sx_ops[(i+1)%size]+
                            sy_ops[i]@sy_ops[(i+1)%size]+
                            sz_ops[i]@sz_ops[(i+1)%size])
        
        hamiltonian += pair_hamiltonian
    return hamiltonian

def generate_hamiltonian_dense(size):
    sx_ops, sy_ops, sz_ops = create_operators_dense(size)

    dimension = 2**size
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)

    for i in range(size):
        pair_hamiltonian = (np.dot(sx_ops[i], sx_ops[(i+1)%size])+
                            np.dot(sy_ops[i], sy_ops[(i+1)%size])+
                            np.dot(sz_ops[i], sz_ops[(i+1)%size]))

        hamiltonian+=pair_hamiltonian
    return hamiltonian

def sector_states(n_particles, n_up):
    return [index for index in range(2**n_particles) if index.bit_count()==n_up]

def hamiltonian_middle_dense(n_particles):

    indices = sector_states(n_particles, ceil(n_particles/2))
    states = []

    for index in indices:
        bitstring = format(index, "b")
        bitstring = bitstring.rjust(n_particles, "0")
        states.append(bitstring)

    hamiltonian = np.zeros((len(states), len(states)))
    index_tracker = {state:index for index, state in enumerate(states)}

    for index, state in enumerate(states):
        #print("\r"+f"Calculating row {index+1}/{len(states)}.", end="")
        for spin in range(n_particles):
            spin_1 = state[spin]
            spin_2 = state[(spin+1)%len(state)]
            if spin_1 == spin_2:
                hamiltonian[state, index_tracker[state]]+=0.25
            else:
                hamiltonian[state, index_tracker[state]]-=0.25

                new_state = list(state)
                new_state[spin] = spin_2
                new_state[(spin+1)%n_particles] = spin_1
                new_state = "".join(new_state)

                hamiltonian[index, index_tracker[new_state]]+=0.5

    return hamiltonian

def hamiltonian_middle_sparse(n_particles):

    indices = sector_states(n_particles, ceil(n_particles/2))

    states = []
    for index in indices:
        bitstring = format(index, "b")
        bitstring = bitstring.rjust(n_particles, "0")
        states.append(bitstring)
    
    hamiltonian = sparse.lil_matrix((len(states), len(states)), dtype=float)
    index_tracker = {state:index for index, state in enumerate(states)}

    for index, state in enumerate(states):
        #print("\r"+f"Calculating row {index+1}/{len(states)}.", end="")
        for spin in range(n_particles):
            spin_1 = state[spin]
            spin_2 = state[(spin+1)%n_particles]
            if spin_1 == spin_2:
                hamiltonian[index, index_tracker[state]]+=0.25
            else:
                hamiltonian[index, index_tracker[state]]-=0.25
                
                new_state = list(state)
                new_state[spin] = spin_2
                new_state[(spin+1)%n_particles] = spin_1
                new_state = "".join(new_state)

                hamiltonian[index, index_tracker[new_state]]+=0.5

    return hamiltonian

def hamiltonian_middle_sparse_bit(n_particles):
    
    states = sector_states(n_particles, ceil(n_particles/2))
    hamiltonian = sparse.lil_matrix((len(states), len(states)), dtype=float)
    index_tracker = {state:index for index, state in enumerate(states)}

    for index, state in enumerate(states):
        #print("\r"+f"Calculating row {index+1}/{len(states)}.", end="")
        for spin in range(n_particles):
            bit1 = 2**spin
            bit2 = 2**((spin+1)%n_particles)
            mask = bit1 + bit2
            if (state & mask) == bit1 or (state&mask) == bit2:
                hamiltonian[index, index_tracker[state]]-=0.25
                hamiltonian[index, index_tracker[state ^ mask]]+=0.5
            else:
                hamiltonian[index, index_tracker[state]]+=0.25

    return hamiltonian

def plot_table(matrix, title):
    fig, ax = plt.subplots()
    norm = plt.Normalize(-1, 1)
    if np.all(matrix == np.conj(matrix)):
        matrix = matrix.real
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

# Generating Hamiltonians
#start_sparse = perf_counter()
#for i in range(2, 23+1):
#    hamiltonian_sparse = generate_hamiltonian_sparse(i)
    #sparse.save_npz(f"operators\h_{i}_sparse", hamiltonian_sparse)
#end_sparse = perf_counter()
#print("\n"+f"It took {end_sparse-start_sparse:.2f}s to generate all 23 sparse matrices."+"\n")

#start_dense = perf_counter()
#for i in range(2, 12+1):
#    hamiltonian_dense = generate_hamiltonian_dense(i)
    #np.save(f"operators\h_{i}_dense", hamiltonian_dense)
#end_dense = perf_counter()
#print(f"It took {end_dense-start_dense:.2f}s to generate all 12 dense matrices."+"\n")

# Loading Hamiltonians
#start_sparse_load = perf_counter()
#for i in range(2, 23+1):
#    sparse.load_npz(f"operators\h_{i}_sparse.npz")
#end_sparse_load = perf_counter()
#print(f"It took {end_sparse_load-start_sparse_load:.2f}s to load all 23 sparse matrices."+"\n")

#start_dense_load = perf_counter()
#for i in range(2, 12+1):
#    np.load(f"operators\h_{i}_dense.npy")
#end_dense_load = perf_counter()
#print(f"It took {end_dense_load-start_dense_load:.2f}s to load all 12 dense matrices."+"\n")

for i in range(2, 26+1):
    print(f"Generating the Hamiltonian for a {i}-particle system.")
    hamiltonian_mid = hamiltonian_middle_sparse_bit(i)
    sparse.save_npz(f"operators\h_{i}_mid_sparse", hamiltonian_mid.tocsr())

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 20:32:22 2026

@author: Maks
"""

from math import ceil, comb
import numpy as np
from scipy import sparse

def sector_states(n_particles, n_up):
    return [index for index in range(2**n_particles) if index.bit_count()==n_up]

def hamiltonian_middle_sparse(n_particles):

    indices = sector_states(n_particles, ceil(n_particles/2))
    print("Indces found.")

    states = []
    for index in indices:
        bitstring = format(index, "b")
        bitstring = bitstring.rjust(n_particles, "0")
        states.append(bitstring)
    print("Indices converted to binary.")
    
    hamiltonian = sparse.csr_matrix((len(states), len(states)))

    for index, state in enumerate(states):
        column_values = {state:0.0 for state in states}
        print(f"Calculating row {index+1}/{len(states)}.")
        for spin in range(len(state)):
            spin_1 = state[spin]
            spin_2 = state[(spin+1)%len(state)]
            if spin_1 == spin_2:
                column_values[state]+=0.25
            else:
                column_values[state]-=0.25
                new_state = list(state)
                new_state[spin] = spin_2
                new_state[(spin+1)%len(state)] = spin_1
                new_state = "".join(new_state)
                column_values[new_state]+=0.5
        hamiltonian[index] = sparse.csr_array(list(column_values.values()))

    return hamiltonian

###############################################################################

#n_particles = 4

#states = sector_states(n_particles, ceil(n_particles/2))

#hamiltonian = np.zeros((len(states), len(states)))

#for index, state in enumerate(states):
#    #print(f"Outer loop triggered with {bin(state)} as state.")
#    temp = state
#    column_values = {state:0.0 for state in states}
#    for spin in range(n_particles):
#        #print(f"Inner loop triggered with {spin} as the spin value.")
#        bit1 = 2**spin
#        bit2 = 2**((spin+1)%n_particles)
#        mask = bit1 + bit2
#        if (temp & mask) == bit1:
#            #print("Encountered 01")
#            column_values[state]-=0.25
#            column_values[state ^ mask]+=0.5
#       elif (temp & mask) == bit2:
#            #print("Encountered 10")
#            column_values[state]-=0.25
#            column_values[state ^ mask]+=0.5
#        else:
#            #print("Encountered 11 or 00")
#            column_values[state]+=0.25
#
#    hamiltonian[index] = np.array(list(column_values.values()))

size_24 = comb(24, 12)
size_25 = comb(25, 13)
size_26 = comb(26, 13)   
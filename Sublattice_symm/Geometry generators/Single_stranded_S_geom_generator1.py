

import numpy as np
import matplotlib.pyplot as plt

#%%
import sys
sys.path.append('C:\\Users\janbr\Downloads\MEP files\Code')    

from N_order_integrator_test import integrate_n_flipaxes

#%%
Tau0 = 4
Lambda0 = 0.5
U0 = 4

x_Length_chain = 6
s_x_indices = [3,4]
s_y_lengths = [1,-1]

unit_length = 1

L_chain_tot = np.sum(np.abs(np.array(s_y_lengths))) + x_Length_chain

def simple_S_geom(x_Length_chain, s_x_indices, s_y_lengths, unit_length = 1.0, Lambda0 = 0.5, Tau0 = 4, U0 = 4):
    
    L_chain_tot = np.sum(np.abs(np.array(s_y_lengths))) + x_Length_chain
    
    #Generating the positions of the chain's sites
    x_chain = unit_length*np.arange(0,L_chain_tot)
    x_store = np.arange(0,s_x_indices[0])
    y_store = np.zeros([s_x_indices[0]])
    
    for i in range(0,len(s_x_indices)):
         #Add vertical chain-segment for each 's'
         sign_y = np.sign(s_y_lengths[i])
         
         x_init, y_init = x_store[-1], y_store[-1]
         x_vals = x_init + np.zeros([np.abs(s_y_lengths[i])])
         if sign_y == 1: #Positive y-increment
             y_vals = y_init + np.arange(0,s_y_lengths[i]) + 1
         if sign_y == -1: #Negative y-increment
             y_vals = y_init + np.flip(np.arange((s_y_lengths[i]),0))
         x_store = np.append(x_store, x_vals) 
         y_store = np.append(y_store, y_vals)
         
         #Add horizontal chain-segment after each 's'
         x_fin, y_fin = x_store[-1], y_store[-1]
         if i < (len(s_x_indices) - 1):
             x_vals = np.arange(s_x_indices[i], s_x_indices[i + 1])
             y_vals = y_fin + np.zeros([s_x_indices[i + 1] - s_x_indices[i]])
         
         if i >= (len(s_x_indices) - 1): #Final horizontal chain-segment of the chain
             x_vals = np.arange(s_x_indices[i], x_Length_chain)
             y_vals = y_fin + np.zeros([x_Length_chain - s_x_indices[i]])
    
         x_store = np.append(x_store, x_vals) 
         y_store = np.append(y_store, y_vals)
    #Scaling to correct dimensions
    x_store = unit_length*x_store
    y_store = unit_length*y_store
    z_store = np.zeros([len(x_store)], dtype = 'complex128')
    Pos_store = np.array([x_store, y_store, z_store])
    
    #Array storing the sites' indices
    Indices_store = np.arange(0, L_chain_tot)
    
    #SOI (NNN) connections
    Conns_1, Conns_2 = Indices_store[:-2], Indices_store[2:]
    Conns_inb = Indices_store[1:-1]
    Conns_SOI = np.array([Conns_1, Conns_inb, Conns_2])
    
    #Tunnel-coupling (NN) connections
    Conns_1_Tau, Conns_2_Tau = Indices_store[:-1], Indices_store[1:]
    Conns_Tau = np.array([Conns_1_Tau, Conns_2_Tau])
    
    #Position differences in order to compute the SOI-vectors v_{i,i+2} = 1j*Lambda0*(d_{i,i+1} x d_{i+1,i+2})
    Pos_diff1 = Pos_store[:,Conns_SOI[1]] - Pos_store[:,Conns_SOI[0]]
    Pos_diff2 = Pos_store[:,Conns_SOI[2]] - Pos_store[:,Conns_SOI[1]]
    print(Pos_diff2)
    plt.plot(x_store, y_store, '-o');plt.xlabel('x');plt.ylabel('y');plt.grid();plt.show()
    
    #Normalizing position-differences
    Pos_diff1_len = np.sqrt(np.sum(Pos_diff1**2, axis = 0))
    Pos_diff2_len = np.sqrt(np.sum(Pos_diff2**2, axis = 0))
    
    Pos_diff1_norm = 1/Pos_diff1_len*Pos_diff1
    Pos_diff2_norm = 1/Pos_diff2_len*Pos_diff2
    
    SOI_vecs = 1j*Lambda0*np.cross(Pos_diff1_norm, Pos_diff2_norm, axisa = 0, axisb = 0, axisc = 0)
    
    #Combining the geometrical part (SOI-vectors) with the Pauli-matrices
    Pauli0 = np.array([[1,0],[0,1]], dtype = 'complex128')
    Pauli_vec = np.zeros([3,2,2], dtype = 'complex128')
    Pauli_vec[0] = np.array([[0,1],[1,0]])
    Pauli_vec[1] = np.array([[0,-1j],[1j,0]])
    Pauli_vec[2] = np.array([[1,0],[0,-1]])
    
    SOI_vecs_exp = np.moveaxis(SOI_vecs, 0, -1)
    
    for i in range(0,2):
        SOI_vecs_exp = np.repeat(np.expand_dims(SOI_vecs_exp, axis = -1), axis = -1, repeats = 2)
    
    SOI_arrS = np.sum(SOI_vecs_exp*Pauli_vec, axis = -3)
    
    #Generating the tunnel-coupling and SOI parts of the Hamiltonian
    SOI_tot = np.zeros([L_chain_tot, 2]*2, dtype = 'complex128')
    Tau_tot = np.zeros([L_chain_tot, 2]*2, dtype = 'complex128')
    U_arr = np.zeros([L_chain_tot, 2]*2, dtype = 'complex128')
    
    SOI_tot[Conns_SOI[0],:,Conns_SOI[2],:] += SOI_arrS
    SOI_tot[Conns_SOI[2],:,Conns_SOI[0],:] += np.swapaxes(np.conj(SOI_arrS), -1, -2)
    
    Tau_tot[Conns_Tau[0],:,Conns_Tau[1],:] += Tau0*Pauli0
    Tau_tot[Conns_Tau[1],:,Conns_Tau[0],:] += np.swapaxes(np.conj(Tau0*Pauli0), -1 ,-2)

    U_arr[np.arange(0,L_chain_tot),:,np.arange(0,L_chain_tot),:] = U0*Pauli_vec[0]
    U_arr_res = np.reshape(U_arr, [2*L_chain_tot]*2)
    return Tau_tot, SOI_tot, U_arr_res

Tau_tot, SOI_tot, U_arr_res = simple_S_geom(x_Length_chain, s_x_indices, s_y_lengths, unit_length = unit_length, Lambda0 = Lambda0, Tau0 = Tau0, U0 = U0)

H_tot = Tau_tot + SOI_tot
SOI_tot_res = np.reshape(SOI_tot, [2*L_chain_tot, 2*L_chain_tot])
Tau_tot_res = np.reshape(Tau_tot, [2*L_chain_tot, 2*L_chain_tot])

H_tot_res = np.reshape(H_tot, [2*L_chain_tot, 2*L_chain_tot])
plt.imshow(np.real(U_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(U_arr_res));plt.colorbar();plt.show()
plt.imshow(np.real(H_tot_res));plt.colorbar();plt.show()
plt.imshow(np.imag(H_tot_res));plt.colorbar();plt.show()
#%%

conns_lead_segL = [0]
conns_lead_segR = [-1]

Gamma_down0_L = 0.75
Gamma_up0_L = 0.25

Gamma_down0_R = 0.5
Gamma_up0_R = 0.5


def simple_S_lead(x_Length_chain, s_x_indices, s_y_lengths, conns_lead, Gamma_up0, Gamma_down0):
    
    L_chain_tot = np.sum(np.abs(np.array(s_y_lengths))) + x_Length_chain
    
    Index_store = np.arange(0,L_chain_tot)
    
    Gamma_lead = np.zeros([L_chain_tot,2]*2, dtype = 'complex128')
    for i in range(0,len(conns_lead)):
        conns_i = conns_lead[i]
        Gamma_lead[conns_i,0,conns_i,0] = Gamma_down0
        Gamma_lead[conns_i,1,conns_i,1] = Gamma_up0
    Gamma_lead_res = np.reshape(Gamma_lead, [2*L_chain_tot]*2)
    return Gamma_lead_res

def plot_simple_S(x_Length_chain, s_x_indices, s_y_lengths, conns_lead):
    
    L_chain_tot = np.sum(np.abs(np.array(s_y_lengths))) + x_Length_chain
    
    #Generating the positions of the chain's sites
    x_chain = unit_length*np.arange(0,L_chain_tot)
    x_store = np.arange(0,s_x_indices[0])
    y_store = np.zeros([s_x_indices[0]])
    
    for i in range(0,len(s_x_indices)):
         #Add vertical chain-segment for each 's'
         sign_y = np.sign(s_y_lengths[i])
         
         x_init, y_init = x_store[-1], y_store[-1]
         x_vals = x_init + np.zeros([np.abs(s_y_lengths[i])])
         if sign_y == 1: #Positive y-increment
             y_vals = y_init + np.arange(0,s_y_lengths[i]) + 1
         if sign_y == -1: #Negative y-increment
             y_vals = y_init + np.flip(np.arange((s_y_lengths[i]),0))
         x_store = np.append(x_store, x_vals) 
         y_store = np.append(y_store, y_vals)
         
         #Add horizontal chain-segment after each 's'
         x_fin, y_fin = x_store[-1], y_store[-1]
         if i < (len(s_x_indices) - 1):
             x_vals = np.arange(s_x_indices[i], s_x_indices[i + 1])
             y_vals = y_fin + np.zeros([s_x_indices[i + 1] - s_x_indices[i]])
         
         if i >= (len(s_x_indices) - 1): #Final horizontal chain-segment of the chain
             x_vals = np.arange(s_x_indices[i], x_Length_chain)
             y_vals = y_fin + np.zeros([x_Length_chain - s_x_indices[i]])
    
         x_store = np.append(x_store, x_vals) 
         y_store = np.append(y_store, y_vals)
    #Scaling to correct dimensions
    x_store = unit_length*x_store
    y_store = unit_length*y_store
    z_store = np.zeros([len(x_store)], dtype = 'complex128')
    Pos_store = np.array([x_store, y_store, z_store])
    
    Pos_leads = Pos_store[:,conns_lead]
    x_leads = Pos_leads[0]
    y_leads = Pos_leads[1]
    plt.plot(x_store, y_store, '-o');
    plt.plot(x_leads, y_leads, 'x', markersize = 10, color = 'black')
    plt.xlabel('x');plt.ylabel('y');plt.grid();plt.show()
    

simple_S_lead(x_Length_chain, s_x_indices, s_y_lengths, conns_lead_segL, Gamma_up0 = Gamma_up0_L, Gamma_down0 = Gamma_down0_L)
plot_simple_S(x_Length_chain, s_x_indices, s_y_lengths, conns_lead_segL + conns_lead_segR)







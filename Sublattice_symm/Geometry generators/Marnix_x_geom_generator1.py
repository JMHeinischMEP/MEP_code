# -*- coding: utf-8 -*-
"""
Created on Sun May 15 12:22:58 2022

@author: janbr
"""
import numpy as np
import matplotlib.pyplot as plt

#%%


Segment_x, Segment_y = 3, 2
Inter_length = 2 #Note: Inter_length > Segment_x
unit_length = 1



Tau0 = 4
Lambda0 = 0.5
U0 = 4


def s_geom_generate(Segment_x, Segment_y, Inter_length, unit_length = 1, Tau0 = 4, Lambda0 = 0.5, U0 = 4, E_field = np.array([0,1,0]), axes_permute = [1,2,0]):
    Conns_segment = np.zeros([Segment_x, Segment_y]*2, dtype = 'int')
    
    #Indexing of the S-structure
    x_index_arr = np.arange(0,Segment_x)
    x_index = np.repeat(np.expand_dims(x_index_arr, axis = 1), axis = 1, repeats = Segment_y)
    
    y_index_arr = np.arange(0,Segment_y)
    y_index = np.repeat(np.expand_dims(y_index_arr, axis = 0), axis = 0, repeats = Segment_x)
    
    index_store = np.zeros([2, Segment_x, Segment_y], dtype = 'int')
    index_store[0,:,:] = x_index
    index_store[1,:,:] = y_index
    
    #Geometry of the structure
    Pos_store0_2D = unit_length*index_store
    Pos_store1_2D = unit_length*index_store
    
    Pos_store0 = np.zeros([3, Segment_x, Segment_y])
    Pos_store1 = np.zeros([3, Segment_x, Segment_y])
    
    Pos_store0[:2] = Pos_store0_2D
    Pos_store1[:2] = Pos_store1_2D

    
    Pos_store1[0] += unit_length*(Segment_x - Inter_length)
    Pos_store1[1] += unit_length*(Segment_y)

    
    axes_permute_inv = np.argsort(axes_permute)
    plot_axes = np.array(['x','y','z'], dtype = 'str')[axes_permute_inv]
    plot_ax0 = plot_axes[0]; plot_ax1 = plot_axes[1]
    
    plt.plot(Pos_store0[0], Pos_store0[1], 'o', color = 'red')
    plt.plot(Pos_store1[0], Pos_store1[1], 'o', color = 'blue')
    plt.title('S-sturcture positions');plt.xlabel(plot_ax0);plt.ylabel(plot_ax1);plt.show()
    
    print(Pos_store0[:2].shape)
    Pos_store0 = Pos_store0[axes_permute]
    Pos_store1 = Pos_store1[axes_permute]

    #Conns_segment[index_store[0], index_store[1], 1,2].shape
    
    #Connections within each of the two segments
    index_store_00 = index_store[:,:-1,:]
    index_store_01 = index_store[:,1:,:]
    index_store_10 = index_store[:,:,:-1]
    index_store_11 = index_store[:,:,1:]
    
    #Storing the connections as Connsi[:,i,j] = [site1_i, site2_j] in which i,j label the connections
    Conns0 = np.array([index_store_00, index_store_01]) #x-connections
    Conns1 = np.array([index_store_10, index_store_11]) #y-connections
    
    Pos_diff0 = Pos_store0[:, Conns0[0][0], Conns0[0][1]] - Pos_store0[:, Conns0[1][0], Conns0[1][1]]
    Pos_diff1 = Pos_store1[:, Conns1[0][0], Conns1[0][1]] - Pos_store1[:, Conns1[1][0], Conns1[1][1]]
    
    #NN Spin-orbit interactions given by (d_{i,j} x E with E = [0,0,1] = \hat{z})
    SOI_vecs0 = Lambda0*1j*np.cross(Pos_diff0, E_field, axisa = 0, axisc = 0)
    SOI_vecs1 = Lambda0*1j*np.cross(Pos_diff1, E_field, axisa = 0, axisc = 0)
    
    print(SOI_vecs0)
    print(SOI_vecs1)
    #Connections between the two segments
    index_inter_s1 = index_store[:,-Inter_length:,-1]
    index_inter_s2 = index_store[:,:Inter_length,0]
    
    #Storing the connections as Connsi[:,i,j] = [site1_i, site2_j] in which i,j label the connections
    Conns_inter = np.array([index_inter_s1, index_inter_s2]) #Connections between the two segments of the S-structure
    
    Pos_diff_inter = Pos_store0[:,Conns_inter[0][0], Conns_inter[0][1]] - Pos_store1[:,Conns_inter[1][0], Conns_inter[1][1]]#np.swapaxes(np.array([np.array([0,1,0])]*Inter_length), 0,1)#
    print(Pos_store0[:,Conns_inter[0][0], Conns_inter[0][1]] - Pos_store1[:,Conns_inter[1][0], Conns_inter[1][1]])
    
    #NN Spin-orbit interactions given by (d_{i,j} x E with E = [0,0,1] = \hat{z})
    SOI_vecs_inter = Lambda0*1j*np.cross(Pos_diff_inter, E_field, axisa = 0, axisc = 0)
    
    print(SOI_vecs_inter)
    
    #Combining the geometrical factor d_{i,j} x E with the spin-part as (d_{i,j} x E) o Pauli_vector
    Pauli0 = np.array([[1,0],[0,1]], dtype = 'complex128')
    
    Pauli_vec = np.zeros([3,2,2], dtype = 'complex128')
    Pauli_vec[0] = np.array([[0,1],[1,0]])
    Pauli_vec[1] = np.array([[0,-1j],[1j,0]])
    Pauli_vec[2] = np.array([[1,0],[0,-1]])
    
    #Combining for SOI within each segment
    SOI_vecs0_exp = np.moveaxis(SOI_vecs0, 0, -1)
    SOI_vecs1_exp = np.moveaxis(SOI_vecs1, 0, -1)
    for i in range(0,2):
        SOI_vecs0_exp = np.expand_dims(SOI_vecs0_exp, axis = -1)
        SOI_vecs0_exp = np.repeat(SOI_vecs0_exp, axis = -1, repeats = 2)
        SOI_vecs1_exp = np.expand_dims(SOI_vecs1_exp, axis = -1)
        SOI_vecs1_exp = np.repeat(SOI_vecs1_exp, axis = -1, repeats = 2)
    
    SOI_arr0 = SOI_vecs0_exp*Pauli_vec; SOI_arr1 = SOI_vecs1_exp*Pauli_vec
    SOI_arr0 = np.sum(SOI_arr0, axis = -3); SOI_arr1 = np.sum(SOI_arr1, axis = -3)
    
    #Combining for SOI between the segments
    SOI_vecs_inter_exp = np.moveaxis(SOI_vecs_inter, 0, -1)
    for i in range(0,2):
        SOI_vecs_inter_exp = np.expand_dims(SOI_vecs_inter_exp, axis = -1)
        SOI_vecs_inter_exp = np.repeat(SOI_vecs_inter_exp, axis = -1, repeats = 2)
    
    SOI_arr_inter = SOI_vecs_inter_exp*Pauli_vec
    SOI_arr_inter = np.sum(SOI_arr_inter, axis = -3)
    
    
    #Constructing the SOI and tunnel-coupling parts of the Hamiltonian
    SOI_tot = np.zeros([2,Segment_x,Segment_y,2]*2, dtype = 'complex128')
    
    SOI_tot[0,Conns0[0][0],Conns0[0][1],:,0,Conns0[1][0],Conns0[1][1],:] += SOI_arr0
    SOI_tot[0,Conns1[0][0],Conns1[0][1],:,0,Conns1[1][0],Conns1[1][1],:] += SOI_arr1
    SOI_tot[0,Conns0[1][0],Conns0[1][1],:,0,Conns0[0][0],Conns0[0][1],:] += np.conj(np.swapaxes(SOI_arr0, -1, -2))
    SOI_tot[0,Conns1[1][0],Conns1[1][1],:,0,Conns1[0][0],Conns1[0][1],:] += np.conj(np.swapaxes(SOI_arr1, -1, -2))
    
    SOI_tot[1,Conns0[0][0],Conns0[0][1],:,1,Conns0[1][0],Conns0[1][1],:] += SOI_arr0
    SOI_tot[1,Conns1[0][0],Conns1[0][1],:,1,Conns1[1][0],Conns1[1][1],:] += SOI_arr1
    SOI_tot[1,Conns0[1][0],Conns0[1][1],:,1,Conns0[0][0],Conns0[0][1],:] += np.conj(np.swapaxes(SOI_arr0, -1, -2))
    SOI_tot[1,Conns1[1][0],Conns1[1][1],:,1,Conns1[0][0],Conns1[0][1],:] += np.conj(np.swapaxes(SOI_arr1, -1, -2))
    
    
    SOI_tot[0,Conns_inter[0][0],Conns_inter[0][1],:,1,Conns_inter[1][0],Conns_inter[1][1],:] += SOI_arr_inter
    SOI_tot[1,Conns_inter[1][0],Conns_inter[1][1],:,0,Conns_inter[0][0],Conns_inter[0][1],:] += np.conj(np.swapaxes(SOI_arr_inter, -1, -2))
    
    Tau_tot = np.zeros([2,Segment_x,Segment_y,2]*2, dtype = 'complex128')
    
    Tau_tot[0,Conns0[0][0],Conns0[0][1],:,0,Conns0[1][0],Conns0[1][1],:] += Pauli0*Tau0
    Tau_tot[0,Conns1[0][0],Conns1[0][1],:,0,Conns1[1][0],Conns1[1][1],:] += Pauli0*Tau0
    Tau_tot[0,Conns0[1][0],Conns0[1][1],:,0,Conns0[0][0],Conns0[0][1],:] += np.conj(np.swapaxes(Pauli0*Tau0, -1, -2))
    Tau_tot[0,Conns1[1][0],Conns1[1][1],:,0,Conns1[0][0],Conns1[0][1],:] += np.conj(np.swapaxes(Pauli0*Tau0, -1, -2))
    
    Tau_tot[1,Conns0[0][0],Conns0[0][1],:,1,Conns0[1][0],Conns0[1][1],:] += Pauli0*Tau0
    Tau_tot[1,Conns1[0][0],Conns1[0][1],:,1,Conns1[1][0],Conns1[1][1],:] += Pauli0*Tau0
    Tau_tot[1,Conns0[1][0],Conns0[1][1],:,1,Conns0[0][0],Conns0[0][1],:] += np.conj(np.swapaxes(Pauli0*Tau0, -1, -2))
    Tau_tot[1,Conns1[1][0],Conns1[1][1],:,1,Conns1[0][0],Conns1[0][1],:] += np.conj(np.swapaxes(Pauli0*Tau0, -1, -2))
    
    Tau_tot[0,Conns_inter[0][0],Conns_inter[0][1],:,1,Conns_inter[1][0],Conns_inter[1][1],:] += Pauli0*Tau0
    Tau_tot[1,Conns_inter[1][0],Conns_inter[1][1],:,0,Conns_inter[0][0],Conns_inter[0][1],:] += np.conj(np.swapaxes(Pauli0*Tau0, -1, -2))
    
    #On site Coulomb interactions
    U_arr = np.zeros([2*Segment_x*Segment_y,2]*2, dtype = 'complex128')
    
    U_arr[np.arange(0,2*Segment_x*Segment_y),:,np.arange(0,2*Segment_x*Segment_y),:] = U0*Pauli_vec[0]
    U_arr_res = np.reshape(U_arr, [2*Segment_x*Segment_y*2]*2)
    
    return Tau_tot, SOI_tot, U_arr_res

Tau_tot, SOI_tot, U_arr_res = s_geom_generate(Segment_x, Segment_y, Inter_length, unit_length = unit_length, Tau0 = Tau0, Lambda0 = Lambda0, U0 = U0, E_field = np.array([1,0,0]))

H_tot = Tau_tot + SOI_tot
#%%

Tau_arr_res = np.reshape(Tau_tot, [2*Segment_x*Segment_y*2]*2)
SOI_arr_res = np.reshape(SOI_tot, [2*Segment_x*Segment_y*2]*2)
H_arr_res = np.reshape(H_tot, [2*Segment_x*Segment_y*2]*2)

plt.imshow(np.real(U_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(U_arr_res));plt.colorbar();plt.show()

plt.imshow(np.real(Tau_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(Tau_arr_res));plt.colorbar();plt.show()
plt.imshow(np.real(SOI_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(SOI_arr_res));plt.colorbar();plt.show()
plt.imshow(np.real(H_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(H_arr_res));plt.colorbar();plt.show()

plt.hist(np.real(np.linalg.eig(H_arr_res)[0]), bins = 50)
plt.show()


#%%

conns_lead_segL = [(0,0,0),(0,0,1)]
conns_lead_segR = [(1,-1,-1),(1,-1,-2)]
#conns_lead_segR = [(-1,-1),(-1,-2)]

Gamma_up0 = 0.75
Gamma_down0 = 0.25

def s_geom_lead(conns_lead_seg, Gamma_up0, Gamma_down0, Segment_x, Segment_y, Inter_length):
    Gamma_lead = np.zeros([2,Segment_x,Segment_y,2]*2, dtype = 'complex128')
    for i in range(0,len(conns_lead_seg)):
        conns_lead_seg_i = conns_lead_seg[i]
        print(conns_lead_seg_i)
        Gamma_lead[conns_lead_seg_i[0], conns_lead_seg_i[1],conns_lead_seg_i[2],0,conns_lead_seg_i[0], conns_lead_seg_i[1],conns_lead_seg_i[2],0] = Gamma_down0
        Gamma_lead[conns_lead_seg_i[0], conns_lead_seg_i[1],conns_lead_seg_i[2],1,conns_lead_seg_i[0], conns_lead_seg_i[1],conns_lead_seg_i[2],1] = Gamma_up0
        
        
    Gamma_lead_res = np.reshape(Gamma_lead, [2*Segment_x*Segment_y*2]*2)
    return Gamma_lead_res

Gamma_L = s_geom_lead(conns_lead_segL, Gamma_up0, Gamma_down0, Segment_x, Segment_y, Inter_length)
plt.imshow(np.real(Gamma_L));plt.colorbar()
plt.show()

def plot_Sgeom(Segment_x, Segment_y, Inter_length, unit_length = 1, conns_lead_seg = []):
        #Indexing of the S-structure
    x_index_arr = np.arange(0,Segment_x)
    x_index = np.repeat(np.expand_dims(x_index_arr, axis = 1), axis = 1, repeats = Segment_y)
    
    y_index_arr = np.arange(0,Segment_y)
    y_index = np.repeat(np.expand_dims(y_index_arr, axis = 0), axis = 0, repeats = Segment_x)
    
    index_store = np.zeros([2, Segment_x, Segment_y], dtype = 'int')
    index_store[0,:,:] = x_index
    index_store[1,:,:] = y_index
    
    #Geometry of the structure
    Pos_store0 = unit_length*index_store
    Pos_store1 = unit_length*index_store
    Pos_store1[0] += unit_length*(Segment_x - Inter_length)
    Pos_store1[1] += unit_length*(Segment_y)

    Pos_store_tot = np.array([Pos_store0, Pos_store1])
    Pos_store_tot = np.swapaxes(Pos_store_tot, 0, 1)
    print(Pos_store_tot.shape)
    
    plt.plot(Pos_store0[0], Pos_store0[1], 'o', color = 'red')
    plt.plot(Pos_store1[0], Pos_store1[1], 'o', color = 'blue')
    
    if conns_lead_seg != []:
        lead_conns_arr = np.array(conns_lead_seg)
        print(lead_conns_arr.shape)
        lead_conns_pos = Pos_store_tot[:,lead_conns_arr[:,0], lead_conns_arr[:,1], lead_conns_arr[:,2]]
        plt.plot(lead_conns_pos[0], lead_conns_pos[1], 'x', color = 'black', markersize = 10)
    plt.title('S-sturcture positions');plt.xlabel('x');plt.ylabel('y');plt.show()

plot_Sgeom(Segment_x, Segment_y, Inter_length, unit_length = unit_length, conns_lead_seg = conns_lead_segL + conns_lead_segR)
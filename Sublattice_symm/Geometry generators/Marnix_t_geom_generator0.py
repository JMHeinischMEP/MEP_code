# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:56:30 2022

@author: janbr
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

#%%

Segment_x, Segment_y = 4,2
Block_x1, Block_x2 = 1,3 #Note: Inter_length > Segment_x
Block_y = 2
unit_length = 1



Tau0 = 4
Lambda0 = 0.5
U0 = 4


def t_geom_generate(Segment_x, Segment_y, Block_x1, Block_x2, Block_y, unit_length = 1, Tau0 = 4, Lambda0 = 0.5, U0 = 4, E_field = np.array([0,1,0]), axes_permute = [1,2,0]):
    
    #Conns_segment = np.zeros([Segment_x, Segment_y]*2, dtype = 'int')
    
    #Indexing of the segment
    x_index_arr = np.arange(0,Segment_x)
    x_index = np.repeat(np.expand_dims(x_index_arr, axis = 1), axis = 1, repeats = Segment_y)
    
    y_index_arr = np.arange(0,Segment_y)
    y_index = np.repeat(np.expand_dims(y_index_arr, axis = 0), axis = 0, repeats = Segment_x)
    
    #Geometry of the segment
    Pos_store0 = np.zeros([3,Segment_x,Segment_y], dtype = 'complex128')
    Pos_store0[0] = unit_length*x_index
    Pos_store0[1] = unit_length*y_index
    
    index_store = np.zeros([2, Segment_x, Segment_y], dtype = 'int')
    index_store[0,:,:] = x_index
    index_store[1,:,:] = y_index
    
    #Connections within the segment
    index_store_00 = index_store[:,:-1,:]
    index_store_01 = index_store[:,1:,:]
    index_store_10 = index_store[:,:,:-1]
    index_store_11 = index_store[:,:,1:]
    
    #Storing the connections as Connsi[:,i,j] = [site1_i, site2_j] in which i,j label the connections
    Conns0 = np.array([index_store_00, index_store_01]) #x-connections
    Conns1 = np.array([index_store_10, index_store_11]) #y-connections



    #Indexing/positions of the Block-structure
    Block_x0 = Block_x2 - Block_x1
    
    x_index_arrB = np.arange(0, Block_x0)
    x_indexB = np.repeat(np.expand_dims(x_index_arrB, axis = 1), axis = 1, repeats = Block_y)
    
    y_index_arrB = np.arange(0, Block_y)
    y_indexB = np.repeat(np.expand_dims(y_index_arrB, axis = 0), axis = 0, repeats = Block_x0)
    
    index_storeB = np.zeros([2, Block_x0, Block_y], dtype = 'int')
    index_storeB[0] = x_indexB
    index_storeB[1] = y_indexB
    #Positions 
    Pos_storeB = np.zeros([3,Block_x0,Block_y], dtype = 'complex128')
    Pos_storeB[0] = unit_length*x_indexB + Block_x1
    Pos_storeB[1] = unit_length*y_indexB + Segment_y
    
    axes_plot_list = np.array(['x','y','z'])
    axes_plot = axes_plot_list[np.argsort(axes_permute)]
    axes_plot0, axes_plot1 = axes_plot[0], axes_plot[1]
    
    plt.plot(Pos_store0[0], Pos_store0[1], 'o', color = 'red')
    plt.plot(Pos_storeB[0], Pos_storeB[1], 'o', color = 'blue')
    plt.title('T-sturcture positions');plt.xlabel(axes_plot0);plt.ylabel(axes_plot1);plt.show()
    #Connections within the block
    index_storeB_00 = index_storeB[:,:-1,:]
    index_storeB_01 = index_storeB[:,1:,:]
    index_storeB_10 = index_storeB[:,:,:-1]
    index_storeB_11 = index_storeB[:,:,1:]
    
    print(index_storeB_10.shape);print(index_storeB_11.shape)
    
    Conns0_B = np.array([index_storeB_00, index_storeB_01])
    Conns1_B = np.array([index_storeB_10, index_storeB_11])
    
    Pos_store0 = Pos_store0[axes_permute]
    Pos_storeB = Pos_storeB[axes_permute]
    
    Pos_diff0 = Pos_store0[:, Conns0[0][0], Conns0[0][1]] - Pos_store0[:, Conns0[1][0], Conns0[1][1]]
    Pos_diff1 = Pos_store0[:, Conns1[0][0], Conns1[0][1]] - Pos_store0[:, Conns1[1][0], Conns1[1][1]]
    
    print(Pos_diff0.shape);print(Pos_diff1.shape)
    
    Pos_diff0_B = Pos_storeB[:,Conns0_B[0][0], Conns0_B[0][1]] - Pos_storeB[:,Conns0_B[1][0], Conns0_B[1][1]]
    Pos_diff1_B = Pos_storeB[:,Conns1_B[0][0], Conns1_B[0][1]] - Pos_storeB[:,Conns1_B[1][0], Conns1_B[1][1]]
    
    #Interactions between the segment and block
    index_inter_Segm = index_store[:,Block_x1:Block_x2,-1]
    index_inter_Block = index_storeB[:,:,0]
    #Connections between the segment and block
    Conns_inter = np.array([index_inter_Segm, index_inter_Block])
    Pos_diff_inter = Pos_store0[:,index_inter_Segm[0], index_inter_Segm[1]] - Pos_storeB[:,index_inter_Block[0], index_inter_Block[1]]
    
    print(Conns_inter)
    print(Pos_diff_inter)
    print(Pos_store0)
    

    
    #NN Spin-orbit interactions given by (d_{i,j} x E with E = [0,1,0] = \hat{y})
    SOI_vecs0 = Lambda0*1j*np.cross(Pos_diff0, E_field, axisa = 0, axisc = 0)
    SOI_vecs1 = Lambda0*1j*np.cross(Pos_diff1, E_field, axisa = 0, axisc = 0)
    SOI_vecs0_B = Lambda0*1j*np.cross(Pos_diff0_B, E_field, axisa = 0, axisc = 0)
    SOI_vecs1_B = Lambda0*1j*np.cross(Pos_diff1_B, E_field, axisa = 0, axisc = 0)
    SOI_vecs_inter = Lambda0*1j*np.cross(Pos_diff_inter, E_field, axisa = 0, axisc = 0)
    
    #Combining the geometrical factor d_{i,j} x E with the spin-part as (d_{i,j} x E) o Pauli_vector
    Pauli0 = np.array([[1,0],[0,1]], dtype = 'complex128')
    
    Pauli_vec = np.zeros([3,2,2], dtype = 'complex128')
    Pauli_vec[0] = np.array([[0,1],[1,0]])
    Pauli_vec[1] = np.array([[0,-1j],[1j,0]])
    Pauli_vec[2] = np.array([[1,0],[0,-1]])
    
    #Combining for SOI within each segment
    SOI_vecs0_exp = np.moveaxis(SOI_vecs0, 0, -1)
    SOI_vecs1_exp = np.moveaxis(SOI_vecs1, 0, -1)
    SOI_vecs0_B_exp = np.moveaxis(SOI_vecs0_B, 0, -1)
    SOI_vecs1_B_exp = np.moveaxis(SOI_vecs1_B, 0, -1)
    SOI_vecs_inter_exp = np.moveaxis(SOI_vecs_inter, 0, -1)
    
    for i in range(0,2):
        SOI_vecs0_exp = np.expand_dims(SOI_vecs0_exp, axis = -1)
        SOI_vecs0_exp = np.repeat(SOI_vecs0_exp, axis = -1, repeats = 2)
        SOI_vecs1_exp = np.expand_dims(SOI_vecs1_exp, axis = -1)
        SOI_vecs1_exp = np.repeat(SOI_vecs1_exp, axis = -1, repeats = 2)
        SOI_vecs0_B_exp = np.expand_dims(SOI_vecs0_B_exp, axis = -1)
        SOI_vecs0_B_exp = np.repeat(SOI_vecs0_B_exp, axis = -1, repeats = 2) 
        SOI_vecs1_B_exp = np.expand_dims(SOI_vecs1_B_exp, axis = -1)
        SOI_vecs1_B_exp = np.repeat(SOI_vecs1_B_exp, axis = -1, repeats = 2) 
        SOI_vecs_inter_exp = np.expand_dims(SOI_vecs_inter_exp, axis = -1)
        SOI_vecs_inter_exp = np.repeat(SOI_vecs_inter_exp, axis = -1, repeats = 2) 
    
    #Combining the geometrical and spin parts    
    SOI_arr0 = SOI_vecs0_exp*Pauli_vec; SOI_arr0 = np.sum(SOI_arr0, axis = -3)
    SOI_arr1 = SOI_vecs1_exp*Pauli_vec; SOI_arr1 = np.sum(SOI_arr1, axis = -3)    
    SOI_arr0_B = SOI_vecs0_B_exp*Pauli_vec; SOI_arr0_B = np.sum(SOI_arr0_B, axis = -3)  
    SOI_arr1_B = SOI_vecs1_B_exp*Pauli_vec; SOI_arr1_B = np.sum(SOI_arr1_B, axis = -3)  
    SOI_arr_inter = SOI_vecs_inter_exp*Pauli_vec; SOI_arr_inter = np.sum(SOI_arr_inter, axis = -3)  
    
    print(SOI_arr1.shape)
    SOI_tot = np.zeros([Segment_x, Segment_y, Block_x0, Block_y,2]*2, dtype = 'complex128')
    print(SOI_tot.shape)
    
    #SOI arrays
    SOI_tot_res = np.zeros([(Segment_x*Segment_y + Block_x0*Block_y)*2]*2, dtype = 'complex128')
    SOI_tot0 = np.zeros([Segment_x, Segment_y, 2]*2, dtype = 'complex128')
    SOI_tot_B = np.zeros([Block_x0, Block_y, 2]*2, dtype = 'complex128')
    SOI_tot_inter = np.zeros([Segment_x, Segment_y, 2, Block_x0, Block_y, 2], dtype = 'complex128')
    
    #Tau arrays
    Tau_tot_res = np.zeros([(Segment_x*Segment_y + Block_x0*Block_y)*2]*2, dtype = 'complex128')
    Tau_tot0 = np.zeros([Segment_x, Segment_y, 2]*2, dtype = 'complex128')
    Tau_tot_B = np.zeros([Block_x0, Block_y, 2]*2, dtype = 'complex128')
    Tau_tot_inter = np.zeros([Segment_x, Segment_y, 2, Block_x0, Block_y, 2], dtype = 'complex128')
    
    #Coulomb array
    U_tot = np.zeros([Segment_x*Segment_y + Block_x0*Block_y,2]*2, dtype = 'complex128')
    
    #Filling the SOI arrays
    SOI_tot0[Conns0[0][0], Conns0[0][1], :, Conns0[1][0], Conns0[1][1], :] += SOI_arr0
    SOI_tot0[Conns1[0][0], Conns1[0][1], :, Conns1[1][0], Conns1[1][1], :] += SOI_arr1
    SOI_tot_B[Conns0_B[0][0], Conns0_B[0][1], :, Conns0_B[1][0], Conns0_B[1][1], :] += SOI_arr0_B
    SOI_tot_B[Conns1_B[0][0], Conns1_B[0][1], :, Conns1_B[1][0], Conns1_B[1][1], :] += SOI_arr1_B
    SOI_tot_inter[Conns_inter[0][0], Conns_inter[0][1],:,Conns_inter[1][0], Conns_inter[1][1],:] += SOI_arr_inter
    #Combining the SOI arrays
    SOI_tot_res[:Segment_x*Segment_y*2,:Segment_x*Segment_y*2] = np.reshape(SOI_tot0, [Segment_x*Segment_y*2]*2)
    SOI_tot_res[:Segment_x*Segment_y*2,:Segment_x*Segment_y*2] += np.conj(np.transpose(np.reshape(SOI_tot0, [Segment_x*Segment_y*2]*2)))
    SOI_tot_res[:Segment_x*Segment_y*2,Segment_x*Segment_y*2:] = np.reshape(SOI_tot_inter, [Segment_x*Segment_y*2, Block_x0*Block_y*2])
    SOI_tot_res[Segment_x*Segment_y*2:,:Segment_x*Segment_y*2] += np.conj(np.transpose(np.reshape(SOI_tot_inter, [Segment_x*Segment_y*2, Block_x0*Block_y*2])))
    SOI_tot_res[Segment_x*Segment_y*2:,Segment_x*Segment_y*2:] = np.reshape(SOI_tot_B, [Block_x0*Block_y*2]*2)
    SOI_tot_res[Segment_x*Segment_y*2:,Segment_x*Segment_y*2:] += np.conj(np.transpose(np.reshape(SOI_tot_B, [Block_x0*Block_y*2]*2)))
    
    plt.imshow(np.real(SOI_tot_res));plt.colorbar();plt.show()
    plt.imshow(np.imag(SOI_tot_res));plt.colorbar();plt.show()
    
    #Filling the Tau arrays
    Tau_tot0[Conns0[0][0], Conns0[0][1], :, Conns0[1][0], Conns0[1][1], :] += Tau0*Pauli0
    Tau_tot0[Conns1[0][0], Conns1[0][1], :, Conns1[1][0], Conns1[1][1], :] += Tau0*Pauli0
    Tau_tot_B[Conns0_B[0][0], Conns0_B[0][1], :, Conns0_B[1][0], Conns0_B[1][1], :] += Tau0*Pauli0
    Tau_tot_B[Conns1_B[0][0], Conns1_B[0][1], :, Conns1_B[1][0], Conns1_B[1][1], :] += Tau0*Pauli0
    Tau_tot_inter[Conns_inter[0][0], Conns_inter[0][1],:,Conns_inter[1][0], Conns_inter[1][1],:] += Tau0*Pauli0
    #Combining the Tau arrays
    Tau_tot_res[:Segment_x*Segment_y*2,:Segment_x*Segment_y*2] = np.reshape(Tau_tot0, [Segment_x*Segment_y*2]*2)
    Tau_tot_res[:Segment_x*Segment_y*2,:Segment_x*Segment_y*2] += np.conj(np.transpose(np.reshape(Tau_tot0, [Segment_x*Segment_y*2]*2)))
    Tau_tot_res[:Segment_x*Segment_y*2,Segment_x*Segment_y*2:] = np.reshape(Tau_tot_inter, [Segment_x*Segment_y*2, Block_x0*Block_y*2])
    Tau_tot_res[Segment_x*Segment_y*2:,:Segment_x*Segment_y*2] += np.conj(np.transpose(np.reshape(Tau_tot_inter, [Segment_x*Segment_y*2, Block_x0*Block_y*2])))
    Tau_tot_res[Segment_x*Segment_y*2:,Segment_x*Segment_y*2:] = np.reshape(Tau_tot_B, [Block_x0*Block_y*2]*2)
    Tau_tot_res[Segment_x*Segment_y*2:,Segment_x*Segment_y*2:] += np.conj(np.transpose(np.reshape(Tau_tot_B, [Block_x0*Block_y*2]*2)))

    plt.imshow(np.real(Tau_tot_res));plt.colorbar();plt.show()
    plt.imshow(np.imag(Tau_tot_res));plt.colorbar();plt.show()
    
    #Filling the Coulomb array
    U_tot[np.arange(0,Segment_x*Segment_y + Block_x0*Block_y),:,np.arange(0,Segment_x*Segment_y + Block_x0*Block_y),:] = Pauli_vec[0]*U0
    U_tot_res = np.reshape(U_tot, [(Segment_x*Segment_y + Block_x0*Block_y)*2]*2)
    plt.imshow(np.real(U_tot_res));plt.colorbar();plt.show()
    plt.imshow(np.imag(U_tot_res));plt.colorbar();plt.show()
    return Tau_tot_res, SOI_tot_res, U_tot_res

    


Tau_tot_res, SOI_tot_res, U_tot_res = t_geom_generate(Segment_x, Segment_y, Block_x1, Block_x2, Block_y, unit_length = unit_length, Tau0 = Tau0, Lambda0 = Lambda0, U0 = U0)
H_tot_res = Tau_tot_res + SOI_tot_res

plt.hist(np.real(np.linalg.eig(H_tot_res)[0]), bins = 100);

#%%
conns_lead_seg = [(0,0),(0,1),(-1,-1),(-1,-2)]
Gamma_up0, Gamma_down0 = 0.25, 0.75

def t_geom_leads(conns_lead_seg, Gamma_up0, Gamma_down0, Segment_x, Segment_y, Block_x1, Block_x2, Block_y, unit_length = 1):
    conns_lead_arr = np.array(conns_lead_seg)
    #Conns_segment = np.zeros([Segment_x, Segment_y]*2, dtype = 'int')
    
    #Indexing of the segment
    x_index_arr = np.arange(0,Segment_x)
    x_index = np.repeat(np.expand_dims(x_index_arr, axis = 1), axis = 1, repeats = Segment_y)
    
    y_index_arr = np.arange(0,Segment_y)
    y_index = np.repeat(np.expand_dims(y_index_arr, axis = 0), axis = 0, repeats = Segment_x)
    
    #Geometry of the segment
    Pos_store0 = np.zeros([2,Segment_x,Segment_y], dtype = 'complex128')
    Pos_store0[0] = unit_length*x_index
    Pos_store0[1] = unit_length*y_index

    plt.plot(Pos_store0[0], Pos_store0[1], 'o', color = 'red')
    plt.plot(Pos_store0[0,conns_lead_arr[:,0],conns_lead_arr[:,1]],Pos_store0[1,conns_lead_arr[:,0],conns_lead_arr[:,1]],'x', markersize = 10, color = 'black')
    plt.xlabel('x');plt.ylabel('y');plt.title('Segment + connections to leads');plt.grid();plt.show()
    Block_x0 = Block_x2 - Block_x1
    
    Gamma_tot = np.zeros([(Segment_x*Segment_y + Block_x0*Block_y)*2]*2, dtype = 'complex128')
    Gamma_arr = np.zeros([Segment_x,Segment_y,2]*2, dtype = 'complex128')
    Gamma_arr[conns_lead_arr[:,0],conns_lead_arr[:,1],0,conns_lead_arr[:,0],conns_lead_arr[:,1],0] = Gamma_down0
    Gamma_arr[conns_lead_arr[:,0],conns_lead_arr[:,1],1,conns_lead_arr[:,0],conns_lead_arr[:,1],1] = Gamma_up0
    
    Gamma_tot[:Segment_x*Segment_y*2,:Segment_x*Segment_y*2] = np.reshape(Gamma_arr, [Segment_x*Segment_y*2]*2)
    plt.imshow(np.real(Gamma_tot));plt.colorbar();plt.show()
    plt.imshow(np.imag(Gamma_tot));plt.colorbar();plt.show()
    return Gamma_tot


Gamma_tot = t_geom_leads(conns_lead_seg, Gamma_up0, Gamma_down0, Segment_x, Segment_y, Block_x1, Block_x2, Block_y, unit_length = unit_length)
    
    
    





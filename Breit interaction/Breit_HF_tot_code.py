# -*- coding: utf-8 -*-
"""
Created on Sat May  7 17:38:04 2022

@author: janbr
"""
import numpy as np
import matplotlib.pyplot as plt

#%%
import sys
sys.path.append('C:\\Users\janbr\Downloads\MEP files\Code')    

from N_order_integrator_test import integrate_n_flipaxes
from Hexagon_construct_functions_only import index_convert, rev_index_convert, generate_connections, Atom_mult_pos
#%%

"""
Defining functions for obtaining matrices which represent the Breit, Coulomb and single-body interactions,
for self-consistently obtaining the occupations in the molecule and computing the current-characteristics
"""


"""
Breit interaction
"""

def get_breit_array(Pos_helix0, N_sites):
    """
    Function which generates an array containing the Breit factors array U_{j(is,i's')}. 
    which is represented by the array Breit_array_tot1, as U_{j(is,i's')} = Breit_arr_tot1[j,i,i',s,'s]

        Input:
            Pos_helix0 : (array, shape = [3,Molecule-length]) Array storing the positions of the sites of the helix molecule. Pos_helix0[i,j] gives the i-th coordinate of the j-th site.
            N_sites : (integer) Number of sites in the molecule
        Returns:
            Breit_array_tot1 : (array, shape = [N_sites,N_sites,N_sites,2,2]) Array containing the Breit factors array U_{j(is,i's')} such that as U_{j(is,i's')} = Breit_arr_tot1[j,i,i',s,'s]
    """
    #Creating an array of shape [3,N_sites,N_sites,N_sites] in which the positions are repeated (such that R_store0[3,i,a,b] = Pos_A_org[3,i] for all a,b in [0, N_sites-1])
    R_store0 = Pos_helix0[:,:N_sites]
    R_store0 = np.repeat(np.expand_dims(R_store0, axis = 2), axis = 2, repeats = N_sites)
    R_store0 = np.repeat(np.expand_dims(R_store0, axis = 3), axis = 3, repeats = N_sites)
    
    #np.cross(R_store0, R_store0, axisa = 0, axisb = 0).shape
    
    #Swapping indices
    R_store_index_0 = np.copy(R_store0)
    R_store_index_1 = np.swapaxes(R_store0, 2, 1)
    R_store_index_2 = np.swapaxes(R_store0, 3, 1)
    
    #Distance between site j and the midpoint of sites i and i', given by |r_j - 1/2*(r_i' + r_i)|
    Mean_pos1 = R_store_index_0 - 1/2*(R_store_index_1 + R_store_index_2)
    Mean_dist1 = np.sqrt(np.sum(Mean_pos1**2, axis = 0))
    Mean_dist1_zeros = 1*(Mean_dist1 == 0)
    Mean_dist1 += Mean_dist1_zeros #Adding 1 to the zero diagonals to avoid infinite values by dividing through zero values
    
    #Cross product between r_{ii'} and r_{i'j}, given by (r_i - r_i') x (r_i' - r_j)
    Cross_diff_Breit1 = np.cross(R_store_index_1 - R_store_index_2, R_store_index_2  - R_store_index_0, axisa = 0, axisb = 0, axisc = 0)
    
    #Distance |r_i' - r_i| for normalization of r_{ii'}
    Abs_dist_12 = np.sqrt(np.sum((R_store_index_1 - R_store_index_2)**2, axis = 0)) 
    Diags = np.repeat(np.expand_dims(np.eye(N_sites, N_sites, k = 0), axis = 0), axis = 0, repeats = N_sites)
    Avoid_diags = 1 - Diags
    Abs_dist_12_diags = Abs_dist_12 + Diags #Adding 1 to the zero diagonals to avoid infinite values by dividing through diagonals
    
    #Only including NN and NNN interactions
    #Indices distinct and nonzero (k!=0, l!=0, k!=l)
    #For any (k,l) there has to exist an (l,k) in order for the Hamiltonian to be Hermitian
    Breit_conns = [(1,2),(2,1),(-1,-2),(-2,-1),(1,-1),(-1,1)]
    
    Inter_conns = np.zeros([N_sites,N_sites,N_sites])
    Fill_diag = np.arange(0,N_sites)
    
    for i in range(0,len(Breit_conns)):
        B_conns1, B_conns2 = Breit_conns[i]
        Max_c, Min_c = np.max(Breit_conns[i]), np.min(Breit_conns[i])
        #Bound array such that indices in Fill_diag_i + B_conns1,2 are all positive and strictly below N_sties
        Fill_diag_i = Fill_diag[(Min_c < 0)*np.abs(Min_c):N_sites - (Max_c > 0)*Max_c] #In case of negative index, bound array from below by smallest index. In case of positive index, bound array from above by largest index
        print(Fill_diag_i)
        Inter_conns[Fill_diag_i, Fill_diag_i + B_conns1, Fill_diag_i + B_conns2] = 1
    
    
    #Combining geometrical factors
    Breit_factor1 = Avoid_diags*1/Abs_dist_12_diags*Cross_diff_Breit1
    Breit_factor1 = Inter_conns*Breit_factor1
    Breit_factor_tot1 = (1 - Mean_dist1_zeros)*1/(Mean_dist1**3)*Breit_factor1
    
    #Combining spatial and spin parts of the Breit interaction
    Breit_factor_tot1 = np.repeat(np.expand_dims(Breit_factor_tot1, axis = 4), axis = 4, repeats = 2)
    Breit_factor_tot1 = np.repeat(np.expand_dims(Breit_factor_tot1, axis = 5), axis = 5, repeats = 2)
    
    Breit_factor_tot1 = np.moveaxis(Breit_factor_tot1, 0, -1)
    
    Pauli_vec_Breit = np.zeros([2,2,3], dtype = 'complex128')
    Pauli_vec_Breit[:,:,0] = np.array([[0,1],[1,0]])
    Pauli_vec_Breit[:,:,1] = np.array([[0,-1j],[1j,0]])
    Pauli_vec_Breit[:,:,2] = np.array([[1,0],[0,-1]])
    
    Breit_array_tot1 = 1j*np.sum(Breit_factor_tot1*Pauli_vec_Breit, axis = -1)
    return Breit_array_tot1


def get_U_Breit_matrices(Breit_array_tot1):
    """
    Function for reshaping the Breit factors such that they are in a convenient shape to produce the Breit matrices U_1,2,3,4 with the array storing the molecule's occupations
        Input:
            Breit_array_tot1 : (array, shape = [N_sites,N_sites,N_sites,2,2]) Array containing the Breit factors array U_{j(is,i's')} such that as U_{j(is,i's')} = Breit_arr_tot1[j,i,i',s,'s]
        Returns:
            U1_Breit_arr, ... : (array, shape depdendent on matrix) Arrays which are used to obtain the Breit matrices U_1,2,3,4 from the occupations in the molecule
    """
    U1_Breit_arr = np.swapaxes(Breit_array_tot1, -3, -2) 
    #U1 indices [j,i,s,i',s'] (shape = [N_sites,N_sites,2,N_sites,2])
    
    U2_Breit_arr = np.moveaxis(Breit_array_tot1, -2, 0) 
    U2_Breit_arr = np.moveaxis(U2_Breit_arr, -1, 1)
    U2_Breit_arr = np.moveaxis(U2_Breit_arr, 2, -1) 
    #U2 indices [s,s'i,i',j] (shape = [2,2,N_sites,N_sites,N_sites])
    
    U3_Breit_arr = 2*np.moveaxis(Breit_array_tot1, 0, 2) #Factor 2 according to the definition of U_{j,j'(is,s')}
    #U3 indices [i,i'j,s,s'] (shape = [N_sites,N_sites,N_sites,2,2])
    
    U4_Breit_arr = 2*Breit_array_tot1 #Factor 2 according to the definition of U_{j,j'(is,s')}
    U4_Breit_arr = np.moveaxis(U4_Breit_arr, -2, 0)
    U4_Breit_arr = np.moveaxis(U4_Breit_arr, -1, 1) 
    #U4 indices [s,s',j,i,i'] (shape = [2,2,N_sites,N_sites,N_sites])
    return U1_Breit_arr, U2_Breit_arr, U3_Breit_arr, U4_Breit_arr


def get_U_Breit(n_arr_test):
    """
    Function for generating the Breit contribution in the Hartree-Fock EOM from the Breit arrays and occupations
        Input:
            n_arr_test : (array, shape = [N_sites,2,N_sites,2]) Array containing the molecule's occupation, in which n_arr_test[i,s,i',s'] = <c_is^{+} c_i's'>
        Returns:
            U_Breit_tot : (array, shape = [N_sites,2,N_sites,2]) This is involved in the Breit contribution in the Hartree-Fock EOM. This term is matrix-multiplied with the Green's function as U_Breit_tot*G to produce the Breit-contribution
    """
    #For each Breit matrix U_1,2,3,4, the occupations in the molecule are reshaped to be combined with these matrices 
    
    #Array U1
    U1_n_arr = n_arr_test #U1 indices [i,s,i',s']
    
    U1_arr_tot0 = U1_Breit_arr*U1_n_arr
    for i in range(0,4):
        U1_arr_tot0 = np.sum(U1_arr_tot0, -1)
    U1_arr_tot = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    U1_arr_tot[np.arange(0,N_sites),0,np.arange(0,N_sites),0] = U1_arr_tot0
    U1_arr_tot[np.arange(0,N_sites),1,np.arange(0,N_sites),1] = U1_arr_tot0
    
    #Array U2
    U2_n_arr = n_arr_test[np.arange(0,N_sites),:,np.arange(0,N_sites),:][:,[0,1],[0,1]]
    U2_n_arr = np.sum(U2_n_arr, axis = 1) #U2 indices [i]
    
    U2_arr_tot = np.sum(U2_Breit_arr*U2_n_arr, axis = -1) 
    U2_arr_tot = np.moveaxis(U2_arr_tot, 0,2)
    U2_arr_tot = np.moveaxis(U2_arr_tot, 0,-1)
    
    #Array U3
    U3_n_arr = n_arr_test[np.arange(0,N_sites),:,np.arange(0,N_sites),:] #U3 indices [i,s,s']
    U3_arr_tot0 = U3_Breit_arr*U3_n_arr
    for i in range(0,3):
        U3_arr_tot0 = np.sum(U3_arr_tot0, axis = -1)
    U3_arr_tot = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    U3_arr_tot[:,0,:,0] = U3_arr_tot0
    U3_arr_tot[:,1,:,1] = U3_arr_tot0
    
    #Array U4    
    U4_n_arr = n_arr_test[:,[0,1],:,[0,1]] 
    U4_n_arr = np.sum(U4_n_arr, axis = 0) #U4 indices [i,i']
    U4_arr_tot0 = U4_Breit_arr*U4_n_arr
    for i in range(0,2):
        U4_arr_tot0 = np.sum(U4_arr_tot0, axis = -1)
    U4_arr_tot0 = np.moveaxis(U4_arr_tot0, -1, 0 )
    U4_arr_tot = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    U4_arr_tot[np.arange(0,N_sites),:,np.arange(0,N_sites),:] = U4_arr_tot0
    
    U_Breit_tot = alpha_Breit*(U1_arr_tot + U2_arr_tot + U3_arr_tot + U4_arr_tot)
    return U_Breit_tot


"""
Matrices for single-body and Coulomb interactions in the molecule
"""

def get_mol_matrices(E_onsite0, Tau0, Lambda0, U0, Cross_diff, N_sites):
    E_site = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
    E_vals = np.array([E_onsite0]*N_sites)
    
    Tau_links = link_generator(N_sites)
    Tau_vals = 4*np.ones([len(Tau_links)], dtype = 'complex128')
    Tau_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
    
    for i in range(0,N_sites):
        E_site[i,[0,1],i,[0,1]] = E_vals[i]
    
    """
    Tunnel-couplings
    """
    
    for i in range(0,len(Tau_links)):
        Tau_i = Tau_vals[i] #+ np.random.random(size = 1)
        Tau_arr[Tau_links[i][0],[0,1],Tau_links[i][1],[0,1]] = Tau_i
        Tau_arr[Tau_links[i][1],[0,1],Tau_links[i][0],[0,1]] = np.conj(Tau_i)
    
    
    """
    Spin-Orbit Interaction
    """
    Pauli_vec = np.zeros([2,2,3], dtype = 'complex128')
    Pauli_vec[:,:,0] = np.array([[0,1],[1,0]])
    Pauli_vec[:,:,1] = np.array([[0,-1j],[1j,0]])
    Pauli_vec[:,:,2] = np.array([[1,0],[0,-1]])
    
    
    SOI_links = SOI_link_generator(N_sites)
    SOI_arr = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    SOI_vecs = Lambda0*Cross_diff[:N_sites]# [0.2*np.array([1,1,1])]*len(SOI_links)
    
    for i in range(0,len(SOI_links)):
        SOI_sum = np.sum(Pauli_vec*SOI_vecs[i], axis = 2)
        SOI_arr[SOI_links[i][0],:,SOI_links[i][1],:] = 1j*SOI_sum
        SOI_arr[SOI_links[i][1],:,SOI_links[i][0],:] = np.conj(np.transpose(1j*SOI_sum))
    #plt.imshow(np.real(SOI_arr_res));plt.colorbar();plt.show()
    #plt.imshow(np.imag(SOI_arr_res));plt.colorbar()

    
    """
    Capacitive interactions
    """
    U_onsite_vals = np.array([U0]*N_sites)
    
    UC_links = link_generator(N_sites)
    UC_vals = np.array([0.]*len(UC_links))
    
    U_tot_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
    #UC_tot_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
    U_Diag_tot_arr = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    for i in range(0,N_sites):
        U_Diag_tot_arr[i,[0,1],i,[1,0]] = U_onsite_vals[i] #Since U_is,is = 0, representing the unphysical self-interactions. U_is,i-s =/= 0
        
    for i in range(0,len(UC_links)):
        U_tot_arr[UC_links[i][0],:,UC_links[i][1],:] = UC_vals[i]
        U_tot_arr[UC_links[i][1],:,UC_links[i][0],:] = np.conj(UC_vals[i])
    
    
    U_onsite_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
    for i in range(0,N_sites):
        U_tot_arr[i,[0,1],i,[0,1]] = U_Diag_tot_arr[i,[0,1],i,[1,0]]
    
    U_tot_arr_res = np.reshape(U_tot_arr, newshape = [2*N_sites, 2*N_sites]) #Reshaping for matrix-multiplication with n
    
    Id_res = np.eye(N_sites*2,N_sites*2,k=0,dtype = 'complex128')
    Id = np.reshape(Id_res, newshape = [N_sites, 2, N_sites, 2])
    
    E_site_res = np.reshape(E_site, newshape = [2*N_sites, 2*N_sites])
    #U_onsite_arr_res = np.reshape(U_onsite_arr, newshape = [2*N_sites, 2*N_sites])
    U_onsite_arr_res = np.reshape(U_Diag_tot_arr, newshape = [2*N_sites, 2*N_sites])
    Tau_arr_res = np.reshape(Tau_arr, newshape = [2*N_sites, 2*N_sites])
    SOI_arr_res = np.reshape(SOI_arr, newshape = [2*N_sites, 2*N_sites])
    
    return Id_res, E_site_res, Tau_arr_res, SOI_arr_res, U_onsite_arr_res


def get_n_arr_res(Occ):
    print(Occ.shape)
    Occ_diag = Occ[np.arange(0,N_sites),:,np.arange(0,N_sites),:]
    Occ_down = Occ_diag[:,0,0]; Occ_up = Occ_diag[:,1,1]
    
    n_arr = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    n_arr[np.arange(0,N_sites),0,np.arange(0,N_sites),0] = Occ_up
    n_arr[np.arange(0,N_sites),1,np.arange(0,N_sites),1] = Occ_down
    
    n_arr_res = np.reshape(n_arr, newshape = [2*N_sites, 2*N_sites])
    return n_arr_res

def get_UC_arr_res(Occ):
    n_arr_2 = np.reshape(Occ, [N_sites*2,N_sites*2])[np.arange(0,N_sites*2), np.arange(0,N_sites*2)]
    UC_arr_res = np.diag(np.dot(UC_tot_arr_res, n_arr_2))
    return UC_arr_res

def link_generator(N):
    link_arr = []
    for i in range(0,N-1):
        link_arr.append((i,i+1))
    return link_arr


def SOI_link_generator(N):
    link_arr = []
    for i in range(0,N-2):
        link_arr.append((i,i+2))
    return link_arr


def get_spin_proj(N_sites):
    #Defining spin-projection matrices for computation of spin-dependent transmission (T_uu, T_ud, ...)
    Spin_proj_down = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    Spin_proj_up = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    
    Spin_proj_down[:,0,:,0] = np.eye(N_sites,N_sites, k=0)
    Spin_proj_up[:,1,:,1] = np.eye(N_sites,N_sites, k=0)
    return Spin_proj_down, Spin_proj_up


"""
Couplings between the molecule and leads
"""


def get_lead_couplings(Gamma_L0_down, Gamma_L0_up, Gamma_R0_down, Gamma_R0_up, Lead_connect_L, Lead_connect_R):
    #Gamma is indepdendent of energy as the wide-band limit is considered
    
    Gamma_L_down = np.zeros([N_sites,N_sites], dtype = 'complex128')
    Gamma_L_up = np.zeros([N_sites,N_sites], dtype = 'complex128')
    
    for i in range(0,len(Lead_connect_L)): #Assuming Gamma's are equal for all connections between lead and molecule
        Gamma_L_down[Lead_connect_L[i],Lead_connect_L[i]] = Gamma_L0_down
        Gamma_L_up[Lead_connect_L[i],Lead_connect_L[i]] = Gamma_L0_up
    
    
    Gamma_L_tot = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    Gamma_L_tot[:,0,:,0] = Gamma_L_down; Gamma_L_tot[:,1,:,1] = Gamma_L_up
    
    
    Gamma_R_down = np.zeros([N_sites,N_sites], dtype = 'complex128')
    Gamma_R_up = np.zeros([N_sites,N_sites], dtype = 'complex128')
    
    for i in range(0,len(Lead_connect_R)):
        Gamma_R_down[Lead_connect_R[i],Lead_connect_R[i]] = Gamma_R0_down
        Gamma_R_up[Lead_connect_R[i],Lead_connect_R[i]] = Gamma_R0_up
    
    Gamma_R_tot = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    Gamma_R_tot[:,0,:,0] = Gamma_R_down; Gamma_R_tot[:,1,:,1] = Gamma_R_up
    
    return Gamma_L_tot, Gamma_R_tot

"""
Self-consistent determination of the molecule's occupations
"""


def GF_ret(E, n_arr_vec, U_array_Breit0_res):# Occ_res):
    Sigma_C_res = np.diag(np.dot(U_onsite_arr_res, n_arr_vec))
    GF_res = np.linalg.inv(E*Id_res - E_site_res - Tau_arr_res - SOI_arr_res - Sigma_C_res - Sigma_tot_res - U_array_Breit0_res)
    return GF_res
#plt.imshow(np.real(GF_ret(0, 0.5*np.ones([2*N_sites], dtype = 'complex128'))))


def FD_dist(E, mu, beta=1):
    return 1/(np.exp((E - mu)*beta) + 1)


def get_GF_tot(e_arr, mu_L, mu_R, n_arr_vec, Occ_res, beta = 1):
    G_less_store = np.zeros([len(e_arr), N_sites, 2, N_sites, 2], dtype = 'complex128')
    
    Occ_shape0 = np.reshape(Occ_res, [N_sites,2,N_sites,2])
    U_array_Breit0 = get_U_Breit(Occ_shape0)
    U_array_Breit0_res = np.reshape(U_array_Breit0, [2*N_sites,2*N_sites])
    
    for i in range(0,len(e_arr)):
        E = e_arr[i]
        
        Sigma_less_res = 1j*(Gamma_L_res*FD_dist(E,mu_L,beta) + Gamma_R_res*FD_dist(E,mu_R,beta))
        G_ret_res = GF_ret(E, n_arr_vec, U_array_Breit0_res)
        G_adv_res = np.conj(np.transpose(G_ret_res))
        
        G_less_res = np.dot(G_ret_res, np.dot(Sigma_less_res, G_adv_res))
        G_less = np.reshape(G_less_res, [N_sites,2,N_sites,2])
        G_less_store[i] = G_less
    return G_less_store

#get_GF_tot(np.linspace(-15,15,10000),-1,1,np.diag(0.5*np.ones([2*N_sites], dtype = 'complex128')))

def get_Occ(G_less_store, e_arr):
    delta_E = np.abs(e_arr[1] - e_arr[0])
    Occ_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
    Occ_arr = 1/(2*np.pi)*integrate_n_flipaxes(-1j*G_less_store, 3, delta_E)
    return Occ_arr

def get_Occ_step(e_arr, mu_L, mu_R, Occ, beta = 1):
    Occ_res = np.reshape(Occ, [2*N_sites,2*N_sites])
    n_arr_vec = Occ_res[np.arange(0,2*N_sites),np.arange(0,2*N_sites)]
    G_less_store = get_GF_tot(e_arr, mu_L, mu_R, n_arr_vec, Occ_res, beta)
    Occ_arr = get_Occ(G_less_store, e_arr)
    return Occ_arr

def get_Occ_SC(e_arr, mu_L, mu_R, Occ_init, n_iter, beta = 1):
    for j in range(0,n_iter):
        if j==0:
            Occ_i = get_Occ_step(e_arr, mu_L, mu_R, Occ_init, beta)
            #Occ_i = get_Occ_step(np.linspace(-20,-10,60000), mu_L, mu_R, Occ_init, beta)  + get_Occ_step(np.linspace(-10,0,60000), mu_L, mu_R, Occ_init, beta) + get_Occ_step(np.linspace(0,10,60000), mu_L, mu_R, Occ_init, beta)  + get_Occ_step(np.linspace(10,20,60000), mu_L, mu_R, Occ_init, beta)
        if j!=0:
            Occ_i = get_Occ_step(e_arr, mu_L, mu_R, Occ_i, beta)
            #Occ_i = get_Occ_step(np.linspace(-20,-10,60000), mu_L, mu_R, Occ_i, beta)  + get_Occ_step(np.linspace(-10,0,60000), mu_L, mu_R, Occ_i, beta) + get_Occ_step(np.linspace(0,10,60000), mu_L, mu_R, Occ_i, beta)  + get_Occ_step(np.linspace(10,20,60000), mu_L, mu_R, Occ_i, beta)
        print(np.reshape(Occ_i, [2*N_sites, 2*N_sites])[np.arange(0,2*N_sites),np.arange(0,2*N_sites)])
    return Occ_i

#get_Occ_step(np.linspace(-15,15,10000),-1,1,np.reshape(np.diag(0.5*np.ones([2*N_sites], dtype = 'complex128')), [N_sites,2,N_sites,2]))

def Occ_sweep_V(mu0, V_arr, e_arr, n_it0, n_it, beta = 1):
    Occ_store = np.zeros([len(V_arr),N_sites,2,N_sites,2], dtype = 'complex128')
    for k in range(0,len(V_arr)):
        mu_L = mu0 + V_arr[k]/2; mu_R = mu0 - V_arr[k]/2
        if k == 0:
            Occ_res = np.diag(0.5*np.ones([2*N_sites], dtype = 'complex128'))
            Occ_i = np.reshape(Occ_res, [N_sites,2,N_sites,2])
            Occ_i = get_Occ_SC(e_arr, mu_L, mu_R, Occ_i, n_it0, beta=beta)
        if k!=0:
            Occ_i = get_Occ_SC(e_arr, mu_L, mu_R, Occ_i, n_it, beta=beta)
        Occ_store[k] = Occ_i
        print(k)
    return Occ_store



"""
Current characteristics
"""

def IV_generate(mu0, V_arr, e_arr_p, beta, Occ_store):
    """
    Computes the currents over a leads kept at chemical potential mu0 over a bias-voltage range specified by V_arr
    from the occupation and off-site capacitive arrays n_arr_sweep_res and UC_arr_sweep_res    
    n_arr_sweep_res[i], UC_arr_sweep_res[i] correspond to the voltage V_arr[i]
    
    """
    
    I_store = np.zeros([len(V_arr)], dtype = 'complex128')
    
    for j in range(0,len(V_arr)):
       # print(V_arr[j])
        
        FD_diff_arr = FD_dist(e_arr_p, mu0 + V_arr[j]/2, beta) - FD_dist(e_arr_p, mu0 - V_arr[j]/2, beta)
        print(j)
       # print(np.max(np.abs(FD_diff_arr)))
        Occ_j = Occ_store[j]
        Occ_res_j = np.reshape(Occ_j, [2*N_sites,2*N_sites])
        n_arr_vec_j = Occ_res_j[np.arange(0,2*N_sites),np.arange(0,2*N_sites)]
        
        U_array_Breit0_j = get_U_Breit(Occ_j)
        U_array_Breit0_res_j = np.reshape(U_array_Breit0_j, [2*N_sites,2*N_sites])
        
       # n_arr_res_j = get_n_arr_res(Occ_j)
       # UC_arr_res_j = get_UC_arr_res(Occ_j)
        
        Transm_arr_down = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_arr_up = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_arr_du = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_arr_ud = np.zeros(len(e_arr_p), dtype = 'complex128')
        
        for i in range(len(e_arr_p)):
            G_ret_res = GF_ret(e_arr_p[i], n_arr_vec_j, U_array_Breit0_res_j)#G_ret_store[i]#GF_ret(e_arr[i] + 0.0000001j,H0,Sigma_L_res + Sigma_R_res)
            G_ret = np.reshape(G_ret_res, [N_sites,2,N_sites,2])
            
            G_adv_res = np.transpose(np.conj(G_ret_res))
            G_adv = np.reshape(G_adv_res, [N_sites,2,N_sites,2])

            
            Ti_down = np.trace(np.dot(Gamma_L_res,np.dot(Spin_res_down,np.dot(G_ret_res,np.dot(Gamma_R_res,np.dot(Spin_res_down,G_adv_res))))))
            Ti_up = np.trace(np.dot(Gamma_L_res,np.dot(Spin_res_up,np.dot(G_ret_res,np.dot(Gamma_R_res,np.dot(Spin_res_up,G_adv_res))))))
            Ti_du = np.trace(np.dot(Gamma_L_res,np.dot(Spin_res_up,np.dot(G_ret_res,np.dot(Gamma_R_res,np.dot(Spin_res_down,G_adv_res))))))
            Ti_ud = np.trace(np.dot(Gamma_L_res,np.dot(Spin_res_down,np.dot(G_ret_res,np.dot(Gamma_R_res,np.dot(Spin_res_up,G_adv_res))))))
            
            Transm_arr_down[i] = Ti_down
            Transm_arr_up[i] = Ti_up
            Transm_arr_du[i] = Ti_du
            Transm_arr_ud[i] = Ti_ud

        Transm_tot = Transm_arr_up + Transm_arr_du + Transm_arr_ud + Transm_arr_down
        Spin_Pol = (Transm_arr_up + (Transm_arr_du - Transm_arr_ud) - Transm_arr_down)/Transm_tot
        I_store[j] = np.abs(e_arr_p[1] - e_arr_p[0])*np.sum(Transm_tot*FD_diff_arr)

    return I_store, Transm_tot, Spin_Pol


#%%
"""
Generating positions of helicene molecule
"""
N_sites = 8 #Number of sites of the molecule

N_t = 25
I_C = index_convert(N_t)
revI_C = rev_index_convert(N_t)

Conn_H = generate_connections(N_t)
revI_C[:,Conn_H][:,0,0]
Pos_A = Atom_mult_pos(revI_C, 1)

Pos_A_org = Pos_A[:,I_C] 
#Pos_A_org[0] = np.copy(-Pos_A_org[0])

DiffPos = np.diff(Pos_A_org[:,:,0], axis = 1)
DiffPos_len = np.sqrt(np.sum(DiffPos**2, axis=0))
DiffPos_norm = DiffPos/DiffPos_len
#Cross_diff =np.cross(DiffPos_norm[:,1:], np.array([1,0,1]), axisa = 0, axisb = 0)
Cross_diff = np.cross(DiffPos_norm[:,1:], DiffPos_norm[:,:-1], axisa = 0, axisb = 0)
#Cross_diff = np.flip(Cross_diff, axis = 0)#%%



#%%

"""
Creating an array Breit_arr_tot1 such that
the Breit interactions array U_{j(is,i's')} is represented as U_{j(is,i's')} = Breit_arr_tot1[j,i,i',s,'s]

Breit_arr_tot1.shape = [N_sites, N_sites, N_sites, 2, 2]
"""

#Positions of the (inner) helix
Pos_helix0 = Pos_A_org[:,:,0]

#Breit interaction strength
alpha_Breit = 0.05 


Breit_array_tot1 = get_breit_array(Pos_helix0, N_sites)

#Creating the Breit matrices U_1,2,3,4
U1_Breit_arr, U2_Breit_arr, U3_Breit_arr, U4_Breit_arr = get_U_Breit_matrices(Breit_array_tot1)

#n_arr_test = Occ_store[0]
#Testing for a given set of occupations
n_arr_test = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
n_arr_test[np.arange(0,N_sites),0,np.arange(0,N_sites),0] = 0.75
n_arr_test[np.arange(0,N_sites),1,np.arange(0,N_sites),1] = 0.25

#Defining a function to obtain the Hartree-Fock Breit matrix from the occupations <c_is^+ c_i's'>
U_Breit_tot = get_U_Breit(n_arr_test)

plt.imshow(np.real(np.reshape(U_Breit_tot, [2*N_sites,2*N_sites])));plt.colorbar();plt.show()
plt.imshow(np.imag(np.reshape(U_Breit_tot, [2*N_sites,2*N_sites])));plt.colorbar();plt.show()

plt.imshow(np.real(np.reshape(U_Breit_tot, [2*N_sites,2*N_sites])) - np.transpose(np.real(np.reshape(U_Breit_tot, [2*N_sites,2*N_sites]))));plt.colorbar();plt.show()
plt.imshow(np.imag(np.reshape(U_Breit_tot, [2*N_sites,2*N_sites])) + np.transpose(np.imag(np.reshape(U_Breit_tot, [2*N_sites,2*N_sites]))));plt.colorbar();plt.show()




#%%

"""
Single/many-body interactions in the molecule
"""
#Approx. realistic params for carbon: Tau0 = 2.4, Lambda0 = 0.005, U0 = 10 (U0 is approximately halved if screening is present)

E_onsite0 = 0 #Onsite energies
Tau0 = 4 #Tunneling coupling strength
Lambda0 = 0.5 #Spin-orbit coupling strength
U0 = 4 #Onsite Coulomb interaction strength


Id_res, E_site_res, Tau_arr_res, SOI_arr_res, U_onsite_arr_res = get_mol_matrices(E_onsite0, Tau0, Lambda0, U0, Cross_diff, N_sites)

plt.imshow(np.real(Id_res));plt.colorbar();plt.title('Identity matrix');plt.show()
plt.imshow(np.real(E_site_res));plt.colorbar();plt.title('Onsite energies');plt.show()
plt.imshow(np.real(Tau_arr_res));plt.colorbar();plt.title('Tunnel couplings');plt.show()
plt.imshow(np.real(SOI_arr_res));plt.colorbar();plt.title('Re(Spin-orbit interactions)');plt.show()
plt.imshow(np.imag(SOI_arr_res));plt.colorbar();plt.title('Im(Spin-orbit interactions)');plt.show()
plt.imshow(np.real(U_onsite_arr_res));plt.colorbar();plt.title('Coulmb interactions');plt.show()
#%%


"""
Couplings with the leads
"""

#Sites which are coupled with the left & right leads
Lead_connect_L = [0,1]
Lead_connect_R = [6,7]

#Coupling strengths between molecule & left lead (Gamma_down, Gamma_up = 0.75, 0.25 => Down magnetized lead, 0.25, 0.75 => Up magnetized lead)
Gamma_L0_down = 0.75
Gamma_L0_up = 0.25
#Coupling strengths between molecule & right lead
Gamma_R0_down = 0.5
Gamma_R0_up = 0.5


Spin_proj_down, Spin_proj_up = get_spin_proj(N_sites)
Gamma_L_tot, Gamma_R_tot = get_lead_couplings(Gamma_L0_down, Gamma_L0_up, Gamma_R0_down, Gamma_R0_up, Lead_connect_L, Lead_connect_R)


Spin_res_down = np.reshape(Spin_proj_down, newshape = [2*N_sites,2*N_sites])
Spin_res_up = np.reshape(Spin_proj_up, newshape = [2*N_sites,2*N_sites])

Gamma_L_res = np.reshape(Gamma_L_tot, newshape = [2*N_sites,2*N_sites])
Gamma_R_res = np.reshape(Gamma_R_tot, newshape = [2*N_sites,2*N_sites])

Sigma_L = -1j/2*Gamma_L_tot
Sigma_R = -1j/2*Gamma_R_tot

Sigma_L_res = np.reshape(Sigma_L,  newshape = [2*N_sites,2*N_sites])
Sigma_R_res = np.reshape(Sigma_R,  newshape = [2*N_sites,2*N_sites])

Sigma_tot_res = Sigma_L_res + Sigma_R_res

plt.imshow(np.real(Gamma_L_res));plt.colorbar();plt.title('Couplings to left lead');plt.show()
plt.imshow(np.real(Gamma_R_res));plt.colorbar();plt.title('Couplings to right lead');plt.show()


#%%

"""
Simulating the molecule:
    Obtaining the occupations for a molecule coupled to leads kept at temperature beta, average chemical potential mu0
    The bias voltage is sweeped over a range stored in V_arr
"""
#Inverse temperature & average chemical potential of the leads
beta = 2.5; mu0 = 0. 
#Energy array over which is integrated to obtain the occupations self-consistently from the lesser Green's function
e_arr = np.linspace(-15,15,15000) 

#Array storing the bias-voltages
V_arr = np.array([2]) 

#Occupations stored in Occ_store such that Occ_store[k,i,s,i',s'] gives the occupations <c_{is}^+ c_{i's'}> at the k-th voltage in V_arr
Occ_store = Occ_sweep_V(mu0, V_arr, e_arr, n_it0 = 20, n_it = 20, beta = beta)
#%%

"""
Obtaining the total current and spin-polarization in the transmission
"""

I_store, Transm_tot, Spin_Pol = IV_generate(mu0, V_arr,e_arr, beta, Occ_store)


plt.plot(V_arr, np.real(I_store))
plt.xlabel('V');plt.ylabel('I')
plt.grid();plt.show()
#%% Comparing two currents

I_store2 = np.copy(I_store)

#%% Comparing two currents

#plt.figure(figsize = (7,5))
plt.plot(V_arr, np.real(I_store-I_store2))
plt.xlabel('V', fontsize = 14);plt.ylabel(r'$\Delta I$', fontsize = 14)
plt.xticks(fontsize = 14);plt.yticks(fontsize = 14)
plt.grid();plt.show()

#plt.figure(figsize = (7,5))
plt.plot(V_arr, 100*np.real(I_store-I_store2)/np.real(I_store + I_store2))
plt.xlabel('V', fontsize = 14);plt.ylabel('MR[%]', fontsize = 14)
plt.xticks(fontsize = 14);plt.yticks(fontsize = 14)
plt.grid();plt.show()

#%%

Occ_store2 = np.copy(Occ_store)
#%%

Occ_store_down, Occ_store_up = Occ_store[:,np.arange(0,N_sites),0,np.arange(0,N_sites),0], Occ_store[:,np.arange(0,N_sites),1,np.arange(0,N_sites),1]
Occ_store2_down, Occ_store2_up = Occ_store2[:,np.arange(0,N_sites),0,np.arange(0,N_sites),0], Occ_store2[:,np.arange(0,N_sites),1,np.arange(0,N_sites),1]

Occ_store_diff1 = Occ_store_down - Occ_store2_up
Occ_store_diff2 = Occ_store_up - Occ_store2_down

for i in range(0,Occ_store.shape[0]):
    plt.plot(Occ_store_diff1[i])
plt.grid();plt.show()


for i in range(0,Occ_store.shape[0]):
    plt.plot(Occ_store_diff2[i])
plt.grid();plt.show()











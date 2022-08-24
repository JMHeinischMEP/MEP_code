# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:52:32 2022

@author: janbr
"""
import numpy as np
import matplotlib.pyplot as plt

#%% Functions for efficiently generating NN/NNN-links

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

#%% Importing higher-order integration function
import sys
sys.path.append('C:\\Users\janbr\Downloads\MEP files\Code')    

from N_order_integrator_test import integrate_n_flipaxes

#%% Generating of the helix' geometry to be used in the SOI-vectors
N_t = 25
I_C = index_convert(N_t)
revI_C = rev_index_convert(N_t)

Conn_H = generate_connections(N_t)
revI_C[:,Conn_H][:,0,0]
Pos_A = Atom_mult_pos(revI_C, 1)

Pos_A_org = Pos_A[:,I_C]
Pos_A_org[0] = np.copy(-Pos_A_org[0])

DiffPos = np.diff(Pos_A_org[:,:,0], axis = 1)
DiffPos_len = np.sqrt(np.sum(DiffPos**2, axis=0))
DiffPos_norm = DiffPos/DiffPos_len
#Cross_diff =np.cross(DiffPos_norm[:,1:], np.array([1,0,1]), axisa = 0, axisb = 0)
Cross_diff = np.cross(DiffPos_norm[:,1:], DiffPos_norm[:,:-1], axisa = 0, axisb = 0)
#Cross_diff = np.flip(Cross_diff, axis = 0)
#%% Creating the matrices which appear in the GF

"""
=> The single-particle on-site energies are stored in the array E_vals, while 
=> the tunnel-coupling strengths are stored in Tau_vals.
=> The vectors stored in the Cross_diff array (defined in the previous section), 
    are scaled by lambda to obtain the SOI vectors in SOI_vecs

=> The on-site (Hubbard) Coulomb-interaction strengths are stored in U_onsite_vals, while 
=> the off-site capacitvie interaction strengths are stored in UC_arr_vals
"""

N_sites = 8
E_site = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
E_vals = np.array([0]*N_sites) #On-site energies 

Tau_links = link_generator(N_sites)
Tau_vals = 4*np.ones([len(Tau_links)], dtype = 'complex128') #Tunnel-coupling strengths
Tau_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')

#Filling the matrix representing the single-particle on-site energies
for i in range(0,N_sites):
    E_site[i,[0,1],i,[0,1]] = E_vals[i]

"""
Tunnel-couplings
=> the tunnel-coupling strengths are stored in Tau_vals.
"""

#Filling the matrix representing the tunnel-couplings
for i in range(0,len(Tau_links)):
    Tau_i = Tau_vals[i] #+ np.random.random(size = 1)
    Tau_arr[Tau_links[i][0],[0,1],Tau_links[i][1],[0,1]] = Tau_i
    Tau_arr[Tau_links[i][1],[0,1],Tau_links[i][0],[0,1]] = np.conj(Tau_i)


"""
Spin-Orbit Interaction
=> The vectors stored in the Cross_diff array (defined in the previous section), 
    are scaled by lambda to obtain the SOI vectors in SOI_vecs
"""
Pauli_vec = np.zeros([2,2,3], dtype = 'complex128')
Pauli_vec[:,:,0] = np.array([[0,1],[1,0]])
Pauli_vec[:,:,1] = np.array([[0,-1j],[1j,0]])
Pauli_vec[:,:,2] = np.array([[1,0],[0,-1]])

#Filling the matrix representing the SOI
SOI_links = SOI_link_generator(N_sites)
SOI_arr = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
SOI_vecs = 0.5*Cross_diff[:N_sites]

for i in range(0,len(SOI_links)):
    SOI_sum = np.sum(Pauli_vec*SOI_vecs[i], axis = 2)
    SOI_arr[SOI_links[i][0],:,SOI_links[i][1],:] = 1j*SOI_sum
    SOI_arr[SOI_links[i][1],:,SOI_links[i][0],:] = np.conj(np.transpose(1j*SOI_sum))
#plt.imshow(np.real(SOI_arr_res));plt.colorbar();plt.show()
#plt.imshow(np.imag(SOI_arr_res));plt.colorbar()
"""
Occupation numbers
"""

def get_n_arr_res(Occ):
    """
    Creates a diagonal array containing the occupations from an array containing all <d^+_is d_i's'>. 
        Input:
            Occ : [N_sites,2,N_sites,2] Array containing the values of <d^+is d_i's'>, which are stored as Occ[i,s,i',s'] = <d^+is d_i's'>
        Returns:
            n_arr_res : [2*N_sites, 2*N_sites] Array containing the occupations <n_is> in the diagonal, which are store according to n_arr_res[2*i+s,2*i+s] = <n_{i, 1-s}> in which s=0 corresponds to down-spin and s=1 corresponds to up-spin
    """
    print(Occ.shape)
    Occ_diag = Occ[np.arange(0,N_sites),:,np.arange(0,N_sites),:]
    Occ_down = Occ_diag[:,0,0]; Occ_up = Occ_diag[:,1,1]
    
    n_arr = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
    n_arr[np.arange(0,N_sites),0,np.arange(0,N_sites),0] = Occ_up
    n_arr[np.arange(0,N_sites),1,np.arange(0,N_sites),1] = Occ_down
    
    n_arr_res = np.reshape(n_arr, newshape = [2*N_sites, 2*N_sites])
    return n_arr_res

def get_UC_arr_res(Occ):
    """
    Creates an array representing the off-site capacitive Coulomb interactions in the molecule.
        Input:
            Occ : [N_sites,2,N_sites,2] Array containing the values of <d^+is d_i's'>, which are stored as Occ[i,s,i',s'] = <d^+is d_i's'>
        Returns:
            UC_arr_res : [2*N_sites, 2*N_sites] Array containing the values of Sum_{i'' \neq i, sigma''}{U_{i sigma, i'' sigma''} <n_{i'' sigma''}>} in the diagonal, which correspond to UC_arr_res[2*i + sigma, 2*i + sigma] in which s=0 corresponds to down-spin and s=1 corresponds to up-spin
    """
    n_arr_2 = np.reshape(Occ, [N_sites*2,N_sites*2])[np.arange(0,N_sites*2), np.arange(0,N_sites*2)]
    UC_arr_res = np.diag(np.dot(UC_tot_arr_res, n_arr_2))
    return UC_arr_res


Occ_arr0_res = np.diag(0.5*np.ones([2*N_sites], dtype = 'complex128'))
Occ_arr0 = np.reshape(Occ_arr0_res, [N_sites,2,N_sites,2])
n_arr0 = get_n_arr_res(Occ_arr0)

"""
Capacitive interactions
=> The on-site (Hubbard) Coulomb-interaction strengths are stored in U_onsite_vals, while 
=> the off-site capacitvie interaction strengths are stored in UC_arr_vals
"""
U_onsite_vals = np.array([10.]*N_sites) #On-site Coulomb interaction strengths

UC_links = link_generator(N_sites) #Links for off-site Coulomb interactions
UC_vals = np.array([0.]*len(UC_links)) #Off-site Coulomb interaction strengths

U_Diag_tot_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
UC_tot_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')

for i in range(0,N_sites):
    U_Diag_tot_arr[i,[0,1],i,[1,0]] = U_onsite_vals[i] #Since U_is,is = 0, representing the unphysical self-interactions. U_is,i-s =/= 0
    
for i in range(0,len(UC_links)):
    UC_tot_arr[UC_links[i][0],:,UC_links[i][1],:] = UC_vals[i]
    UC_tot_arr[UC_links[i][1],:,UC_links[i][0],:] = np.conj(UC_vals[i])

UC_tot_arr_res = np.reshape(UC_tot_arr, newshape = [2*N_sites, 2*N_sites]) #Reshaping for matrix-multiplication with n

U_onsite_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
for i in range(0,N_sites):
    U_onsite_arr[i,[0,1],i,[0,1]] = U_Diag_tot_arr[i,[0,1],i,[1,0]]

#Identity array/matrix
Id_res = np.eye(N_sites*2,N_sites*2,k=0,dtype = 'complex128')
Id = np.reshape(Id_res, newshape = [N_sites, 2, N_sites, 2])

#Reshaping of matrices to 2D arrays in order to perform linear-algebra operations with Numpy
E_site_res = np.reshape(E_site, newshape = [2*N_sites, 2*N_sites])
U_onsite_arr_res = np.reshape(U_onsite_arr, newshape = [2*N_sites, 2*N_sites])
Tau_arr_res = np.reshape(Tau_arr, newshape = [2*N_sites, 2*N_sites])
SOI_arr_res = np.reshape(SOI_arr, newshape = [2*N_sites, 2*N_sites])
#%% Generating the matrices representing the couplings to the leads

Lead_connect_L = [0,1]
Lead_connect_R = [6,7]

"""
In this section, the matrices for the couplings to the lead are generated
=> The couplings for Gamma_{L \sigma} are specified in Gamma_L0_down/up 
=> The couplings for Gamma_{R \sigma} are specified in Gamma_R0_down/up 
"""

#Gamma is indepdendent of energy as the wide-band limit is considered

#Defining spin-projection matrices for computation of spin-dependent transmission (T_uu, T_ud, ...)
Spin_proj_down = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
Spin_proj_up = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')

Spin_proj_down[:,0,:,0] = np.eye(N_sites,N_sites, k=0)
Spin_proj_up[:,1,:,1] = np.eye(N_sites,N_sites, k=0)

Spin_res_down = np.reshape(Spin_proj_down, newshape = [2*N_sites,2*N_sites])
Spin_res_up = np.reshape(Spin_proj_up, newshape = [2*N_sites,2*N_sites])


Gamma_L0_down = 0.75
Gamma_L0_up = 0.25


Gamma_L_down = np.zeros([N_sites,N_sites], dtype = 'complex128')
Gamma_L_up = np.zeros([N_sites,N_sites], dtype = 'complex128')

for i in range(0,len(Lead_connect_L)): #Assuming Gamma's are equal for all connections between lead and molecule
    Gamma_L_down[Lead_connect_L[i],Lead_connect_L[i]] = Gamma_L0_down
    Gamma_L_up[Lead_connect_L[i],Lead_connect_L[i]] = Gamma_L0_up


Gamma_L_tot = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
Gamma_L_tot[:,0,:,0] = Gamma_L_down; Gamma_L_tot[:,1,:,1] = Gamma_L_up
Gamma_L_res = np.reshape(Gamma_L_tot, newshape = [2*N_sites,2*N_sites])

Gamma_R0_down = 0.5
Gamma_R0_up = 0.5

Gamma_R_down = np.zeros([N_sites,N_sites], dtype = 'complex128')
Gamma_R_up = np.zeros([N_sites,N_sites], dtype = 'complex128')

for i in range(0,len(Lead_connect_R)):
    Gamma_R_down[Lead_connect_R[i],Lead_connect_R[i]] = Gamma_R0_down
    Gamma_R_up[Lead_connect_R[i],Lead_connect_R[i]] = Gamma_R0_up

Gamma_R_tot = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
Gamma_R_tot[:,0,:,0] = Gamma_R_down; Gamma_R_tot[:,1,:,1] = Gamma_R_up
Gamma_R_res = np.reshape(Gamma_R_tot, newshape = [2*N_sites,2*N_sites])

Sigma_L = -1j/2*Gamma_L_tot
Sigma_R = -1j/2*Gamma_R_tot

Sigma_L_res = np.reshape(Sigma_L,  newshape = [2*N_sites,2*N_sites])
Sigma_R_res = np.reshape(Sigma_R,  newshape = [2*N_sites,2*N_sites])

Sigma_tot_res = Sigma_L_res + Sigma_R_res

#%% Defining the functions used to self-consistently determine the occupations in the molecule
# For a schematic overview, see the Code_structure_schematic.png image uploaded in the Surfdrive )in the same map as this code)

def generate_GF_ret_res(E, n_arr_res, UC_arr_res):#, E_site_res=E_site_res, Tau_arr_res=Tau_arr_res, SOI_arr_res=SOI_arr_res, Sigma_tot_res=Sigma_tot_res, U_onsite_arr_res=U_onsite_arr_res, UC_arr_res=UC_arr_res, Id_res = Id_res):
    """
    Computes the retarded GF in the 'Hubbard-I'-approximation for a given <n_-sigma> and off-site capacitive array
        Input:
            E : Energy
            n_arr_res : [2*N_sites,2*N_sites] Reshaped array containing occupations <n_-sigma> in the diagonal
        Returns:
            G_ret_res : [2*N_sites,2*N_sites] Reshaped retarded Green's function
    """
    G_num = E*Id_res - E_site_res - np.dot(U_onsite_arr_res,Id_res - n_arr_res)
    G_den = np.dot(E*Id_res - E_site_res -  U_onsite_arr_res, E*Id_res - E_site_res -  UC_arr_res - Tau_arr_res - SOI_arr_res) - np.dot(U_onsite_arr_res, np.dot(n_arr_res, Tau_arr_res + SOI_arr_res))
    
    G_non_int_res = np.dot(np.linalg.inv(G_num), G_den)#np.dot(np.linalg.inv(G_den), G_num)
    G_ret_res = np.linalg.inv(G_non_int_res - Sigma_tot_res)
    return G_ret_res


def FD_dist(E, mu, beta=1):
    """
    Fermi-Dirac distribution
    """
    return 1/(np.exp((E - mu)*beta) + 1)


def get_GF_E_res(e_arr, mu_L, mu_R, n_arr_res, UC_arr_res, beta=1):
    """
    Computes all retarded and lesser GFs for the energies in e_arr
    System is coupled to left & right leads kept at chemical potentials mu_L and mu_R, respectively and at temperature beta
    Array <n-\sigma> and UC_arr_res are the occupation-arrays and non-site Capacitive interaction arrays (both diagonal)
        Input:
            e_arr : 1D Array containing the energies for which the retarded & lesser GFs are determined
            mu_L, mu_R : Chemical potentials of respectively the left and right leads
            n_arr_res : shape = [2*N_sites,2*N_sites] Reshaped array containing occupations <n_-sigma> in the diagonal
            beta : Inverse temperature
        Returns: 
            G_ret_store : shape = [len(e_arr),N_sites,2,N_sites,2] Array containing the retarded GFs at all energies in e_arr
            G_less_store : shape = [len(e_arr),N_sites,2,N_sites,2] Array containing the lesser GFs at all energies in e_arr
    """
    G_less_store = np.zeros([len(e_arr)] + [N_sites,2,N_sites,2],dtype = 'complex128')
    G_ret_store = np.zeros([len(e_arr)] + [N_sites,2,N_sites,2],dtype = 'complex128')
    
    for i in range(0,len(e_arr)):
        E = e_arr[i]
        
        Sigma_less_res = 1j*(Gamma_L_res*FD_dist(E,mu_L,beta) + Gamma_R_res*FD_dist(E,mu_R,beta))
        G_ret_res = generate_GF_ret_res(E, n_arr_res, UC_arr_res)#, E_site_res, Tau_arr_res, SOI_arr_res, Sigma_tot_res, U_onsite_arr_res, UC_arr_res, Id_res = Id_res)
        G_ret = np.reshape(G_ret_res, newshape = [N_sites, 2, N_sites, 2])
        
        G_adv_res = np.conj(np.transpose(G_ret_res))
        G_less_res = np.dot(G_ret_res, np.dot(Sigma_less_res, G_adv_res)) #Keldysh equation
        G_less = np.reshape(G_less_res, newshape = [N_sites, 2, N_sites, 2])
        
        G_ret_store[i] = G_ret
        G_less_store[i] = G_less
    return G_ret_store, G_less_store

#def f_fit_integral(x,A1,b1,c1):# = A
#    return (A1*x**(-np.abs(b1)) + c1)#

def get_occup(G_less_store, e_arr):
    """
    Computes the occupations from the lesser GFs in G_less_store, where the energy of 
    G_less_store[i] corresponds to the energy given by e_arr[i]
        Input:
            G_less_store : shape = [len(e_arr),N_sites,2,N_sites,2] Array containing all lesser GFs at all energies specified in e_arr
            e_arr : 1D array storing all energies, in which e_arr[i] corresponds to G_less_store[i]. Integration is performed over this range, where the order of integration is set in the 2nd argument of the function integrate_n_flipaxes(Integration_arr, order, delta_E)
        Returns:
            Occ_arr : shape = [N_sites,2,N_sites,2] Array containing occupations of the sites
    """
    delta_E = np.abs(e_arr[1] - e_arr[0])
    Occ_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
    
    #Occ_arr = 1/(2*np.pi)*delta_E*np.sum(-1j*G_less_store, axis = 0)
    Occ_arr = 1/(2*np.pi)*integrate_n_flipaxes(-1j*G_less_store, 3, delta_E)
    return Occ_arr



def get_Occ_SC(Occ, e_arr, mu_L, mu_R, beta = 1):
    """
    Function for obtaining the occupations from an initial guess of occupations. 
    Occupations are obtained by obtaining the lesser GF from the retarded (and advanced) GFs from the initial guess, 
    and integrating the lesser GF over all energies
        Input:
            Occ : shape = [N_sites,2,N_sites,2] Initial guess of occupations
            e_arr : 1D array over which is integrated to obtain the final occupations from the lesser GF
            mu_L, mu_R : Chemical potentials of respectively the left and right leads
            beta : Inverse temperature
        Returns:
            Occ_arr : shape = [N_sites,2,N_sites,2]
    """

    n_arr_res = get_n_arr_res(Occ)
    print(n_arr_res[np.arange(0,2*N_sites), np.arange(0,2*N_sites)])
    UC_arr_res = get_UC_arr_res(Occ)
    
    G_ret_store, G_less_store = get_GF_E_res(e_arr, mu_L, mu_R, n_arr_res, UC_arr_res, beta = beta)
    
    plt.plot(np.real(G_less_store[:,0,0,0,0]));plt.plot(np.imag(G_less_store[:,0,0,0,0]));
    plt.show()
    plt.plot(np.real(G_less_store[:,0,1,0,0]));plt.plot(np.imag(G_less_store[:,0,1,0,0]));
    plt.show()
    
    Occ_arr = get_occup(G_less_store, e_arr)
    
    print(Occ_arr[np.arange(0,N_sites),0,np.arange(0,N_sites),0])    
    return (1*Occ_arr + 0*Occ)

def get_Occ_conv(Occ0, e_arr, mu_L, mu_R, n_iter, beta = 1):
    """
    Function for obtaining the occupations of the sites in a self-consistent loop
        Input:
            Occ0 : shape = [N_sites,2,N_sites,2] Initial guess of occupations
            e_arr : 1D array over which is integrated to obtain the final occupations from the lesser GF
            mu_L, mu_R : Chemical potentials of respectively the left and right leads
            n_iter : Number of iterative steps
            beta : Inverse temperature
        Returns
            Occ_i : shape = [N_sites,2,N_sites,2] Array containing the self-consistently obtained occuaptions
    """
    for j in range(0,n_iter):
        if j==0:
            Occ_i = get_Occ_SC(Occ0, e_arr, mu_L, mu_R, beta = beta)
        if j!=0:
            Occ_i = get_Occ_SC(Occ_i, e_arr, mu_L, mu_R, beta = beta)
    return Occ_i


def Occ_V_sweep(mu0, V_arr, e_arr, n_it0, n_it, beta = 1):
    """
    Function for obtaining the occupations over a specified range of bias-voltages
        Input:
            mu0 : Chemical potential of the leads in equilibrium/zero bias-voltage
            V_arr : Array containing the bias-voltages over which the occupations are obtained
            n_it0 : Number of iterative steps for the first bias-voltage
            n_it : Number of iterative steps for voltages after the first bias-voltage
            beta : Inverse temperature
        Returns:
            Occ_store : [len(V_arr), N_sties,2,N_sites,2] Array containing occupations of all sites for each bias-voltages
    """
    Occ_store = np.zeros([len(V_arr), N_sites, 2, N_sites, 2], dtype = 'complex128')
    for k in range(0,len(V_arr)):
        mu_L = mu0 + V_arr[k]/2; mu_R = mu0 - V_arr[k]/2
        if k==0:
            n_iter = n_it0
            Occ_arr0_res = np.diag(0.5*np.ones([2*N_sites], dtype = 'complex128'))
            Occ_arr0 = np.reshape(Occ_arr0_res, [N_sites,2,N_sites,2])
        if k!=0:
            n_iter = n_it
            Occ_arr0 = Occ_i
        
        Occ_i = get_Occ_conv(Occ_arr0, e_arr, mu_L, mu_R, n_iter, beta = beta)
        Occ_store[k] = Occ_i
    return Occ_store
            


#%% Computing the occupations in the molecule for a specified voltage-range
    
"""
The occupations are determined self-consistently in this section
"""

beta = 2.5; mu0 = 0 #Leads' temperature + chem. pot. at zero bias-voltage
e_arr = np.linspace(-15,15,32000) #Energy-range over which is integrated to obtain occupations <n_{i-s}> from the lesser GF
V_arr = np.linspace(-0,0.5,1) #Specified voltage-range

#NOTE: e_arr should have a size which is a multiple of 5. The higher-order integration function integrate_n_flipaxes() 
#only accepts arrays where array.shape[0] = N*(2*P - 1), where P is the integration-order. Since P=3 (3rd order integration) in this code, e_arr should satisfy
# len(e_arr) = N*5
Occ_store = Occ_V_sweep(mu0, V_arr, e_arr, n_it0 = 20, n_it = 20, beta = beta)

#%% For comparison between occupations
Occ_store2 = np.copy(Occ_store)
#%% Comparing occupations & spin-densities in the molecule
plt.imshow(np.real(Occ_store[:,np.arange(0,N_sites),1,np.arange(0,N_sites),1]) - np.real(Occ_store[:,np.arange(0,N_sites),0,np.arange(0,N_sites),0]));plt.colorbar()
plt.show()
plt.imshow(np.real(Occ_store2[:,np.arange(0,N_sites),1,np.arange(0,N_sites),1]) - np.real(Occ_store2[:,np.arange(0,N_sites),0,np.arange(0,N_sites),0]));plt.colorbar()
plt.show()
plt.imshow(np.real(Occ_store[:,np.arange(0,N_sites),1,np.arange(0,N_sites),1]) - np.real(Occ_store[:,np.arange(0,N_sites),0,np.arange(0,N_sites),0]) + (np.real(Occ_store2[:,np.arange(0,N_sites),1,np.arange(0,N_sites),1]) - np.real(Occ_store2[:,np.arange(0,N_sites),0,np.arange(0,N_sites),0])));plt.colorbar()
plt.show()
#%% Computing the current, SPT, etc.


"""
The current, SPT, etc. are computed from the self-consistently obtained occupations (computed in a preceeding section)
"""

def IV_generate(mu0, V_arr, e_arr_p, beta, Occ_store):
    """
    Computes the currents over a leads kept at chemical potential mu0 over a bias-voltage range specified by V_arr
    from the occupations. These occupations should be determined for the same V_arr as used in the self-consistent determination.
    It is desirable to have the energy-array e_arr_p equal to e_arr used in the self-consistent loop, however, using e_arr =/= e_arr_p 
    generates highly comparable/similar results if they are not too different, i.e. approx. same range/resolution
        Input:
            mu0 : Chemical potential of the leads in equilibrium/zero bias-voltage
            V_arr : Array containing the bias-voltages over which the occupations are obtained
            e_arr_p : 1D array over which is integrated to obtain the final occupations from the lesser GF
            beta : Inverse temperature
            Occ_store : [len(V_arr), N_sties,2,N_sites,2] Array containing occupations of all sites for each bias-voltages
        Returns:
            I_store : 1D array of shape [len(V_arr)] storing the currents at voltages specified in V_arr
            I_SD_Store : 1D array of shape [len(V_arr)] storing the differences between the spin-up and -down current at voltages specified in V_arr
            Transm_tot : 1D array of shape [len(e_arr_p)] storing the total transmission at the final voltage V_arr[-1]
            Spin_Pol : 1D array of shape [len(e_arr_p)] storing the SPT (spin-polarization in the transmission) at the final voltage V_arr[-1]
    """
    
    I_store = np.zeros([len(V_arr)], dtype = 'complex128')
    I_SD_store = np.zeros([len(V_arr)], dtype = 'complex128')
    for j in range(0,len(V_arr)):
        FD_diff_arr = FD_dist(e_arr_p, mu0 + V_arr[j]/2, beta) - FD_dist(e_arr_p, mu0 - V_arr[j]/2, beta)
        print(j)

        Occ_j = Occ_store[j]
        
        n_arr_res_j = get_n_arr_res(Occ_j)
        UC_arr_res_j = get_UC_arr_res(Occ_j)
        
        Transm_arr_down = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_arr_up = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_arr_du = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_arr_ud = np.zeros(len(e_arr_p), dtype = 'complex128')
        
        for i in range(len(e_arr_p)):
            G_ret_res = generate_GF_ret_res(e_arr_p[i], n_arr_res_j, UC_arr_res_j)#G_ret_store[i]#GF_ret(e_arr[i] + 0.0000001j,H0,Sigma_L_res + Sigma_R_res)
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
        Transm_diff = (Transm_arr_up + (Transm_arr_du - Transm_arr_ud) - Transm_arr_down)
        Spin_Pol = Transm_diff/Transm_tot
        #I_store[j] = np.abs(e_arr_p[1] - e_arr_p[0])*np.sum(Transm_tot*FD_diff_arr)
        I_store[j] = integrate_n_flipaxes(Transm_tot*FD_diff_arr, 3, np.abs(e_arr_p[1] - e_arr_p[0]))
        I_SD_store[j] = integrate_n_flipaxes(Transm_diff*FD_diff_arr, 3, np.abs(e_arr_p[1] - e_arr_p[0]))
    return I_store, I_SD_store, Transm_tot, Spin_Pol


I_store, I_SD_store, Transm_tot, Spin_Pol = IV_generate(mu0, V_arr,e_arr, beta, Occ_store)

I_SPT_store = I_SD_store/I_store

plt.plot(V_arr, np.real(I_store))

plt.xlabel('V');plt.ylabel('I')
plt.grid();plt.show()

plt.plot(V_arr, np.real(I_SPT_store))
plt.xlabel('V');plt.ylabel('SPT')
plt.grid();plt.show()
#%% For comparison between two currents
"""
The current-values are stored in I_store2 to be compared to the current-values for the opposite magnetization in the leads
"""
I_store2 = np.copy(I_store)
#%% Plotting the current-asymmetry/difference + magnetoresistance MR (in [%])

"""
The currents between opposite magnetizations are compared
"""

plt.figure(figsize = (7,5))
plt.plot(V_arr, np.real(I_store-I_store2))

plt.xlabel('V', fontsize = 16);plt.ylabel(r'$\Delta I$', fontsize = 16)
plt.xticks(fontsize = 14);plt.yticks(fontsize = 14)
plt.grid();plt.show()

plt.figure(figsize = (7,5))
plt.plot(V_arr, 100*np.real(I_store-I_store2)/np.real(I_store + I_store2))

plt.xlabel('V', fontsize = 16);plt.ylabel('MR[%]', fontsize = 16)
plt.xticks(fontsize = 14);plt.yticks(fontsize = 14)
plt.grid();plt.show()

#%%
Occ_store2 = np.copy(Occ_store)
#%%

Occ_store_down, Occ_store_up = Occ_store[:,np.arange(0,N_sites),0,np.arange(0,N_sites),0], Occ_store[:,np.arange(0,N_sites),1,np.arange(0,N_sites),1]
Occ_store2_down, Occ_store2_up = Occ_store2[:,np.arange(0,N_sites),0,np.arange(0,N_sites),0], Occ_store2[:,np.arange(0,N_sites),1,np.arange(0,N_sites),1]

Occ_store_diff1 = Occ_store_down - Occ_store2_down
Occ_store_diff2 = Occ_store_up - Occ_store2_up

for i in range(0,Occ_store.shape[0]):
    plt.plot(Occ_store_diff1[i])
plt.grid();plt.show()

for i in range(0,Occ_store.shape[0]):
    plt.plot(Occ_store_diff2[i])
plt.grid();plt.show()

plt.imshow(np.real(Occ_store_down));plt.colorbar();plt.show()
#%%

I_SD_store2 = np.copy(I_SD_store)
I_SPT_store2 = np.copy(I_SPT_store)
#%%

plt.figure(figsize = (7,5))
plt.plot(V_arr, I_SD_store2 + I_SD_store)
plt.xlabel('V', fontsize = 16);plt.ylabel(r'$\Sigma (I_\uparrow - I_\downarrow)$', fontsize = 16)
plt.xticks(fontsize = 14);plt.yticks(fontsize = 14)
plt.grid();plt.show()

plt.figure(figsize = (7,5))
plt.plot(V_arr, I_SPT_store2 + I_SPT_store)
plt.xlabel('V', fontsize = 16);plt.ylabel(r'$\Sigma (SPT)$', fontsize = 16)
plt.xticks(fontsize = 14);plt.yticks(fontsize = 14)
plt.grid();plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 18:35:16 2022

@author: janbr
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
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

#%%
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
N_sites = 8
E_site = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
E_vals = np.array([0]*N_sites)

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
SOI_vecs = 0.5*Cross_diff[:N_sites]# [0.2*np.array([1,1,1])]*len(SOI_links)

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


Occ_arr0_res = np.diag(0.5*np.ones([2*N_sites], dtype = 'complex128'))
Occ_arr0 = np.reshape(Occ_arr0_res, [N_sites,2,N_sites,2])
n_arr0 = get_n_arr_res(Occ_arr0)

"""
Capacitive interactions
"""
U_onsite_vals = np.array([4.]*N_sites)

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
#%%

Lead_connect_L = [0]
Lead_connect_R = [7]


#Gamma is indepdendent of energy as the wide-band limit is considered

#Defining spin-projection matrices for computation of spin-dependent transmission (T_uu, T_ud, ...)
Spin_proj_down = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')
Spin_proj_up = np.zeros([N_sites,2,N_sites,2], dtype = 'complex128')

Spin_proj_down[:,0,:,0] = np.eye(N_sites,N_sites, k=0)
Spin_proj_up[:,1,:,1] = np.eye(N_sites,N_sites, k=0)

Spin_res_down = np.reshape(Spin_proj_down, newshape = [2*N_sites,2*N_sites])
Spin_res_up = np.reshape(Spin_proj_up, newshape = [2*N_sites,2*N_sites])


Gamma_L0_down = 0.075
Gamma_L0_up = 0.025


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

#%%

def GF_ret(E, n_arr_vec):
    Sigma_C_res = np.diag(np.dot(U_onsite_arr_res, n_arr_vec))
    GF_res = np.linalg.inv(E*Id_res - E_site_res - Tau_arr_res - SOI_arr_res - Sigma_C_res - Sigma_tot_res)
    return GF_res
plt.imshow(np.real(GF_ret(0, 0.5*np.ones([2*N_sites], dtype = 'complex128'))))


def FD_dist(E, mu, beta=1):
    return 1/(np.exp((E - mu)*beta) + 1)


def get_GF_tot(e_arr, mu_L, mu_R, n_arr_vec, beta = 1):
    G_less_store = np.zeros([len(e_arr), N_sites, 2, N_sites, 2], dtype = 'complex128')
    
    for i in range(0,len(e_arr)):
        E = e_arr[i]
        
        Sigma_less_res = 1j*(Gamma_L_res*FD_dist(E,mu_L,beta) + Gamma_R_res*FD_dist(E,mu_R,beta))
        G_ret_res = GF_ret(E, n_arr_vec)
        G_adv_res = np.conj(np.transpose(G_ret_res))
        
        G_less_res = np.dot(G_ret_res, np.dot(Sigma_less_res, G_adv_res))
        G_less = np.reshape(G_less_res, [N_sites,2,N_sites,2])
        G_less_store[i] = G_less
    return G_less_store

#get_GF_tot(np.linspace(-15,15,10000),-1,1,np.diag(0.5*np.ones([2*N_sites], dtype = 'complex128')))

def get_Occ(G_less_store, e_arr):
    delta_E = np.abs(e_arr[1] - e_arr[0])
    Occ_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
    
    Occ_arr = 1/(2*np.pi)*delta_E*np.sum(-1j*G_less_store, axis = 0)

    return Occ_arr

def get_Occ_step(e_arr, mu_L, mu_R, Occ, beta = 1):
    Occ_res = np.reshape(Occ, [2*N_sites,2*N_sites])
    n_arr_vec = Occ_res[np.arange(0,2*N_sites),np.arange(0,2*N_sites)]
    G_less_store = get_GF_tot(e_arr, mu_L, mu_R, n_arr_vec, beta)
    Occ_arr = get_Occ(G_less_store, e_arr)
    return Occ_arr

def get_Occ_SC(e_arr, mu_L, mu_R, Occ_init, n_iter, beta = 1):
    for j in range(0,n_iter):
        if j==0:
            Occ_i = get_Occ_step(e_arr, mu_L, mu_R, Occ_init, beta)
        if j!=0:
            Occ_i = get_Occ_step(e_arr, mu_L, mu_R, Occ_i, beta)
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
    return Occ_store
            
    #%%
Oc0 = np.reshape(np.diag(0.5*np.ones([2*N_sites], dtype = 'complex128')), [N_sites,2,N_sites,2])
Oc = get_Occ_SC(np.linspace(-15,15,10000),1,-1,Oc0, 15, beta = 1)
#%%
beta = 2.5; mu0 = 0
e_arr = np.linspace(-15,15,15000)
V_arr = np.linspace(-1,1,10)

Occ_store = Occ_sweep_V(mu0, V_arr, e_arr, n_it0 = 20, n_it = 20, beta = beta)
#%%

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
        
       # n_arr_res_j = get_n_arr_res(Occ_j)
       # UC_arr_res_j = get_UC_arr_res(Occ_j)
        
        Transm_arr_down = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_arr_up = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_arr_du = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_arr_ud = np.zeros(len(e_arr_p), dtype = 'complex128')
        
        for i in range(len(e_arr_p)):
            G_ret_res = GF_ret(e_arr_p[i], n_arr_vec_j)#G_ret_store[i]#GF_ret(e_arr[i] + 0.0000001j,H0,Sigma_L_res + Sigma_R_res)
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
I_store, Transm_tot, Spin_Pol = IV_generate(mu0, V_arr,e_arr, beta, Occ_store)

plt.plot(V_arr, np.real(I_store))

plt.xlabel('V');plt.ylabel('I')
plt.grid();plt.show()
#%%

I_store2 = np.copy(I_store)

#%%

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

Occ_store_diff1 = Occ_store_down - Occ_store2_up
Occ_store_diff2 = Occ_store_up - Occ_store2_down

for i in range(0,Occ_store.shape[0]):
    plt.plot(Occ_store_diff1[i])
plt.grid();plt.show()


for i in range(0,Occ_store.shape[0]):
    plt.plot(Occ_store_diff2[i])
plt.grid();plt.show()








# -*- coding: utf-8 -*-
"""
Created on Sun May 15 17:44:44 2022

@author: janbr
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
import sys
sys.path.append('C:\\Users\janbr\Downloads\MEP files\Code')    

from N_order_integrator_test import integrate_n_flipaxes
import Marnix_x_geom_generator1 as MXG
import Single_stranded_S_geom_generator1 as SimpleS
import Marnix_t_geom_generator0 as MTG
#%% Single stranded (simple) S-geometry
E0 = 0
Tau0 = 4
Lambda0 = 0.5
U0 = 4


x_Length_chain = 5
s_x_indices = [3]
s_y_lengths = [1]

unit_length = 1

L_chain_tot = np.sum(s_y_lengths) + x_Length_chain
N_sites = np.copy(L_chain_tot)

Id_res = np.eye(2*N_sites, 2*N_sites, k=0, dtype = 'complex128')
E_site_res = E0*Id_res
Tau_tot, SOI_tot, U_onsite_arr_res = SimpleS.simple_S_geom(x_Length_chain, s_x_indices, s_y_lengths, unit_length = unit_length, Lambda0 = Lambda0, Tau0 = Tau0, U0 = U0)

H_tot = Tau_tot + SOI_tot
SOI_arr_res = np.reshape(SOI_tot, [2*L_chain_tot, 2*L_chain_tot])
Tau_arr_res = np.reshape(Tau_tot, [2*L_chain_tot, 2*L_chain_tot])

H_tot_res = np.reshape(H_tot, [2*L_chain_tot, 2*L_chain_tot])

plt.imshow(np.real(U_onsite_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(U_onsite_arr_res));plt.colorbar();plt.show()
plt.imshow(np.real(Tau_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(Tau_arr_res));plt.colorbar();plt.show()
plt.imshow(np.real(SOI_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(SOI_arr_res));plt.colorbar();plt.show()
plt.imshow(np.real(H_tot_res));plt.colorbar();plt.show()
plt.imshow(np.imag(H_tot_res));plt.colorbar();plt.show()
#%% T geometry
E0 = 0
Tau0 = 4
Lambda0 = 0.5
U0 = 4


Segment_x, Segment_y = 4,2
Block_x1, Block_x2 = 1,3 #Note: Inter_length > Segment_x
Block_y = 2
unit_length = 1


N_sites = Segment_x*Segment_y + (Block_x2 - Block_x1)*Block_y

Id_res = np.eye(2*N_sites, 2*N_sites, k=0, dtype = 'complex128')
E_site_res = E0*Id_res
Tau_tot, SOI_tot, U_onsite_arr_res = MTG.t_geom_generate(Segment_x, Segment_y, Block_x1, Block_x2, Block_y, unit_length = 1, Tau0 = 4, Lambda0 = 0.5, U0 = 4)

H_tot = Tau_tot + SOI_tot
SOI_arr_res = np.reshape(SOI_tot, [2*N_sites, 2*N_sites])
Tau_arr_res = np.reshape(Tau_tot, [2*N_sites, 2*N_sites])

H_tot_res = np.reshape(H_tot, [2*N_sites, 2*N_sites])

plt.imshow(np.real(U_onsite_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(U_onsite_arr_res));plt.colorbar();plt.show()
plt.imshow(np.real(Tau_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(Tau_arr_res));plt.colorbar();plt.show()
plt.imshow(np.real(SOI_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(SOI_arr_res));plt.colorbar();plt.show()
plt.imshow(np.real(H_tot_res));plt.colorbar();plt.show()
plt.imshow(np.imag(H_tot_res));plt.colorbar();plt.show()
#%% S-geometry
E0 = 0
Tau0 = 4
Lambda0 = 0.5
U0 = 0.5


Segment_x, Segment_y = 3,2
Inter_length = 2
unit_length = 1

N_sites = 2*Segment_x*Segment_y

Id_res = np.eye(2*N_sites, 2*N_sites, k=0, dtype = 'complex128')
E_site_res = E0*Id_res
Tau_tot, SOI_tot, U_onsite_arr_res = MXG.s_geom_generate(Segment_x, Segment_y, Inter_length, unit_length = unit_length, Tau0 = Tau0, Lambda0 = Lambda0, U0 = U0)

H_tot = Tau_tot + SOI_tot


Tau_arr_res = np.reshape(Tau_tot, [2*Segment_x*Segment_y*2]*2)
SOI_arr_res = np.reshape(SOI_tot, [2*Segment_x*Segment_y*2]*2)
H_arr_res = np.reshape(H_tot, [2*Segment_x*Segment_y*2]*2)

plt.imshow(np.real(U_onsite_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(U_onsite_arr_res));plt.colorbar();plt.show()
plt.imshow(np.real(Tau_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(Tau_arr_res));plt.colorbar();plt.show()
plt.imshow(np.real(SOI_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(SOI_arr_res));plt.colorbar();plt.show()
plt.imshow(np.real(H_arr_res));plt.colorbar();plt.show()
plt.imshow(np.imag(H_arr_res));plt.colorbar();plt.show()

plt.hist(np.real(np.linalg.eig(H_arr_res)[0]), bins = 50)
plt.show()

#%% S geometry

conns_lead_segL = [(0,0,0),(0,0,1)]
conns_lead_segR = [(1,-1,-1),(1,-1,-2)]

Gamma_down0_L = 0.25
Gamma_up0_L = 0.75

Gamma_down0_R = 0.5
Gamma_up0_R = 0.5

Gamma_L_res = MXG.s_geom_lead(conns_lead_segL, Gamma_up0_L, Gamma_down0_L, Segment_x, Segment_y, Inter_length)
Gamma_R_res = MXG.s_geom_lead(conns_lead_segR, Gamma_up0_R, Gamma_down0_R, Segment_x, Segment_y, Inter_length)

Sigma_L = -1j/2*Gamma_L_res
Sigma_R = -1j/2*Gamma_R_res

Sigma_tot_res = Sigma_L + Sigma_R

MXG.plot_Sgeom(Segment_x, Segment_y, Inter_length, unit_length = unit_length, conns_lead_seg = conns_lead_segL + conns_lead_segR)
plt.imshow(np.real(Gamma_L_res));plt.colorbar();plt.show()
plt.imshow(np.real(Gamma_R_res));plt.colorbar();plt.show()

Spin_res_down = np.zeros([2*N_sites,2*N_sites], dtype = 'complex128')
Spin_res_up = np.zeros([2*N_sites,2*N_sites], dtype = 'complex128')

Spin_res_down[np.arange(0,2*N_sites,2), np.arange(0,2*N_sites,2)] = 1
Spin_res_up[np.arange(1,2*N_sites,2), np.arange(1,2*N_sites,2)] = 1
#%% Single stranded (simple) S-geometry
conns_lead_segL = [0,1]
conns_lead_segR = [-1,-2]

Gamma_down0_L = 0.25
Gamma_up0_L = 0.75

Gamma_down0_R = 0.5
Gamma_up0_R = 0.5

Gamma_L_res = SimpleS.simple_S_lead(x_Length_chain, s_x_indices, s_y_lengths, conns_lead_segL, Gamma_up0_L, Gamma_down0_L)
Gamma_R_res = SimpleS.simple_S_lead(x_Length_chain, s_x_indices, s_y_lengths, conns_lead_segR, Gamma_up0_R, Gamma_down0_R)

Sigma_L = -1j/2*Gamma_L_res
Sigma_R = -1j/2*Gamma_R_res

Sigma_tot_res = Sigma_L + Sigma_R

SimpleS.plot_simple_S(x_Length_chain, s_x_indices, s_y_lengths, conns_lead_segL + conns_lead_segR)
plt.imshow(np.real(Gamma_L_res));plt.colorbar();plt.show()
plt.imshow(np.real(Gamma_R_res));plt.colorbar();plt.show()

Spin_res_down = np.zeros([2*N_sites,2*N_sites], dtype = 'complex128')
Spin_res_up = np.zeros([2*N_sites,2*N_sites], dtype = 'complex128')

Spin_res_down[np.arange(0,2*N_sites,2), np.arange(0,2*N_sites,2)] = 1
Spin_res_up[np.arange(1,2*N_sites,2), np.arange(1,2*N_sites,2)] = 1

#%% T-geometry

conns_lead_segL = [(0,0),(0,1)]
conns_lead_segR = [(-1,-1),(-1,-2)]

Gamma_down0_L = 0.75
Gamma_up0_L = 0.25

Gamma_down0_R = 0.5
Gamma_up0_R = 0.5

Gamma_L_res = MTG.t_geom_leads(conns_lead_segL, Gamma_up0_L, Gamma_down0_L, Segment_x, Segment_y, Block_x1, Block_x2, Block_y, unit_length = unit_length)
Gamma_R_res = MTG.t_geom_leads(conns_lead_segR, Gamma_up0_R, Gamma_down0_R, Segment_x, Segment_y, Block_x1, Block_x2, Block_y, unit_length = unit_length)

Sigma_L = -1j/2*Gamma_L_res
Sigma_R = -1j/2*Gamma_R_res

Sigma_tot_res = Sigma_L + Sigma_R

plt.imshow(np.real(Gamma_L_res));plt.colorbar();plt.show()
plt.imshow(np.real(Gamma_R_res));plt.colorbar();plt.show()

Spin_res_down = np.zeros([2*N_sites,2*N_sites], dtype = 'complex128')
Spin_res_up = np.zeros([2*N_sites,2*N_sites], dtype = 'complex128')

Spin_res_down[np.arange(0,2*N_sites,2), np.arange(0,2*N_sites,2)] = 1
Spin_res_up[np.arange(1,2*N_sites,2), np.arange(1,2*N_sites,2)] = 1
 
#%%
def GF_ret(E, n_arr_vec):# Occ_res):
    Sigma_C_res = np.diag(np.dot(U_onsite_arr_res, n_arr_vec))
    
    #Occ_shape0 = np.reshape(Occ_res, [N_sites,2,N_sites,2])
    #U_array_Breit0 = get_U_Breit(Occ_shape0)
    #U_array_Breit0_res = np.reshape(U_array_Breit0, [2*N_sites,2*N_sites])
    GF_res = np.linalg.inv(E*Id_res  - Tau_arr_res - SOI_arr_res - Sigma_C_res - Sigma_tot_res)
    return GF_res
#plt.imshow(np.real(GF_ret(0, 0.5*np.ones([2*N_sites], dtype = 'complex128'))))


def FD_dist(E, mu, beta=1):
    return 1/(np.exp((E - mu)*beta) + 1)


def get_GF_tot(e_arr, mu_L, mu_R, n_arr_vec, Occ_res, beta = 1):
    G_less_store = np.zeros([len(e_arr), N_sites, 2, N_sites, 2], dtype = 'complex128')
    G_ret_store = np.zeros([len(e_arr), 2*N_sites, 2*N_sites], dtype = 'complex128')
    
    for i in range(0,len(e_arr)):
        E = e_arr[i]
        
        Sigma_less_res = 1j*(Gamma_L_res*FD_dist(E,mu_L,beta) + Gamma_R_res*FD_dist(E,mu_R,beta))
        G_ret_res = GF_ret(E, n_arr_vec)
        G_adv_res = np.conj(np.transpose(G_ret_res))
        
        G_less_res = np.dot(G_ret_res, np.dot(Sigma_less_res, G_adv_res))
        G_less = np.reshape(G_less_res, [N_sites,2,N_sites,2])
        G_less_store[i] = G_less
        G_ret_store[i] = G_ret_res
    plt.plot(e_arr, np.real(-1j*G_less_store[:,0,0,0,0]));plt.show()
    G_adv_store = np.swapaxes(np.conj(G_ret_store), -1, -2)
    DOS_tot = np.sum((G_ret_store - G_adv_store)[:,np.arange(0,N_sites),np.arange(0,N_sites)], axis = 1)
    plt.plot(e_arr, 1j*DOS_tot);plt.show()
    return G_less_store

#get_GF_tot(np.linspace(-15,15,10000),-1,1,np.diag(0.5*np.ones([2*N_sites], dtype = 'complex128')))

def get_Occ(G_less_store, e_arr):
    delta_E = np.abs(e_arr[1] - e_arr[0])
    Occ_arr = np.zeros([N_sites, 2, N_sites, 2], dtype = 'complex128')
    
    #Occ_arr = 1/(2*np.pi)*delta_E*np.sum(-1j*G_less_store, axis = 0)
    Occ_arr = 1/(2*np.pi)*integrate_n_flipaxes(-1j*G_less_store, 3, delta_E)
    return Occ_arr

def get_Occ_step(e_arr, mu_L, mu_R, Occ, beta = 1):
    Occ_res = np.reshape(Occ, [2*N_sites,2*N_sites])
    n_arr_vec = Occ_res[np.arange(0,2*N_sites),np.arange(0,2*N_sites)]
    G_less_store = get_GF_tot(e_arr, mu_L, mu_R, n_arr_vec, Occ_res, beta)
    Occ_arr = get_Occ(G_less_store, e_arr)
    return 1*Occ_arr

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
    conv_store = np.zeros([len(V_arr)])
    for k in range(0,len(V_arr)):
        mu_L = mu0 + V_arr[k]/2; mu_R = mu0 - V_arr[k]/2
        if k == 0:
            Occ_res = np.diag(0.5*np.ones([2*N_sites], dtype = 'complex128'))
            Occ_i = np.reshape(Occ_res, [N_sites,2,N_sites,2])
            Occ_i, conv_measure = get_Occ_SC(e_arr, mu_L, mu_R, Occ_i, n_it0, beta=beta)
        if k!=0:
            Occ_i, conv_measure = get_Occ_SC(e_arr, mu_L, mu_R, Occ_i, n_it, beta=beta)
        Occ_store[k] = Occ_i
        conv_store[k] = conv_measure
        print(k)
    return Occ_store, conv_store

def get_Occ_SC(e_arr, mu_L, mu_R, Occ_init, n_iter, beta = 1):
    conv_Bool = True
    conv_measure_store = []
    j = 0
    Occ_i = Occ_init
    Occs_zero = []
    while conv_Bool:
        print(j)
        print(mu_L - mu_R)
        Occ_new = get_Occ_step(e_arr, mu_L, mu_R, Occ_i, beta)
        conv_measure = np.sqrt(np.sum(np.abs(Occ_new - Occ_i)**2))
        conv_Bool = (conv_measure > 10**(-8))
        Occ_i = Occ_new
            #Occ_i = get_Occ_step(np.linspace(-20,-10,60000), mu_L, mu_R, Occ_i, beta)  + get_Occ_step(np.linspace(-10,0,60000), mu_L, mu_R, Occ_i, beta) + get_Occ_step(np.linspace(0,10,60000), mu_L, mu_R, Occ_i, beta)  + get_Occ_step(np.linspace(10,20,60000), mu_L, mu_R, Occ_i, beta)
        print(np.reshape(Occ_i, [2*N_sites, 2*N_sites])[np.arange(0,2*N_sites),np.arange(0,2*N_sites)])
        print(conv_measure)
        conv_measure_store.append(conv_measure)
        Occs_zero.append(Occ_i[2,0,4,1])

        plt.figure(figsize = (5,7));plt.subplot(211)
        plt.plot(conv_measure_store, '-x');plt.grid();plt.ylabel('conv');plt.show()
        plt.figure(figsize = (5,7));plt.subplot(212)
        plt.plot(np.log(np.abs(np.array(conv_measure_store)))/np.log(10), '-x');plt.grid();plt.ylabel(r'$\log_{10}(conv)$');plt.show()
        plt.plot(Occs_zero,'-x');plt.ylabel('Occupations[2,0,4,1]');plt.grid();plt.show()
        if j > 250:
            conv_Bool = False
        j += 1
    return Occ_i, conv_measure
            

#%%
Oc0 = np.reshape(np.diag(0.5*np.ones([2*N_sites], dtype = 'complex128')), [N_sites,2,N_sites,2])
Oc = get_Occ_SC(np.linspace(-15,15,10000),1,-1,Oc0, 15, beta = 1)
#%%
beta = 10; mu0 = 1.5
e_arr = np.linspace(-15,15,30000)
V_arr = np.linspace(-20,20,41)

Occ_store, conv_store = Occ_sweep_V(mu0, V_arr, e_arr, n_it0 = 20, n_it = 20, beta = beta)
#%%

Occ_store2, conv_store2 = np.copy(Occ_store), np.copy(conv_store)
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

    return I_store, Transm_tot, Spin_Pol, Transm_arr_up, Transm_arr_down, Transm_arr_du, Transm_arr_ud


def IV_generate2(mu0, V_arr, e_arr_p, beta, Occ_store):
    """
    Computes the currents over a leads kept at chemical potential mu0 over a bias-voltage range specified by V_arr
    from the occupation and off-site capacitive arrays n_arr_sweep_res and UC_arr_sweep_res    
    n_arr_sweep_res[i], UC_arr_sweep_res[i] correspond to the voltage V_arr[i]
    
    """
    
    I_store = np.zeros([len(V_arr)], dtype = 'complex128')
    
    for j in range(0,len(V_arr)):
       # print(V_arr[j])
        mu_L, mu_R = mu0 + V_arr[j]/2, mu0 - V_arr[j]/2
        print(j)
       # print(np.max(np.abs(FD_diff_arr)))
        Occ_j = Occ_store[j]
        Occ_res_j = np.reshape(Occ_j, [2*N_sites,2*N_sites])
        n_arr_vec_j = Occ_res_j[np.arange(0,2*N_sites),np.arange(0,2*N_sites)]
        
        
        Trace_store = np.zeros(len(e_arr_p), dtype = 'complex128')
        
        for i in range(len(e_arr_p)):
            G_ret_res = GF_ret(e_arr_p[i], n_arr_vec_j)#G_ret_store[i]#GF_ret(e_arr[i] + 0.0000001j,H0,Sigma_L_res + Sigma_R_res)
            G_ret = np.reshape(G_ret_res, [N_sites,2,N_sites,2])
            
            G_adv_res = np.transpose(np.conj(G_ret_res))
            G_adv = np.reshape(G_adv_res, [N_sites,2,N_sites,2])
            
            Sigma_less_res = 1j*(Gamma_L_res*FD_dist(e_arr_p[i],mu_L,beta) + Gamma_R_res*FD_dist(e_arr_p[i],mu_R,beta))
            
            print((G_ret_res - G_adv_res)[0])
            print(1)
            print(1j*np.dot(G_ret_res, np.dot(Gamma_L_res + Gamma_R_res, G_adv_res))[0])
            
            G_less_res = np.dot(G_ret_res, np.dot(Sigma_less_res, G_adv_res))
            Trace_store[i] = np.trace(np.dot(Gamma_L_res - Gamma_R_res, G_less_res)) + np.trace(np.dot(Gamma_L_res*FD_dist(e_arr_p[i],mu_L,beta) - Gamma_R_res*FD_dist(e_arr_p[i],mu_R,beta), G_ret_res - G_adv_res))
        
        #I_store[j] = np.abs(e_arr_p[1] - e_arr_p[0])*np.sum(Transm_tot*FD_diff_arr)
        I_store[j] = 0.5j*integrate_n_flipaxes(Trace_store, 3, np.abs(e_arr_p[1] - e_arr_p[0]))
        Transm_tot, Spin_Pol = 0,0
    return I_store, Transm_tot, Spin_Pol

I_store, Transm_tot, Spin_Pol, Tup, Tdown, Tdu, Tud = IV_generate(mu0, V_arr,e_arr, beta, Occ_store)


plt.plot(V_arr, np.real(I_store))

plt.xlabel('V');plt.ylabel('I')
plt.grid();plt.show()
#%%

I_store2 = np.copy(I_store)
#%%

Occ_store_c = np.copy()
#%%

#plt.figure(figsize = (7,5))
plt.plot(V_arr, np.real(I_store-I_store2), linewidth = 2)

plt.xlabel('V', fontsize = 14);plt.ylabel(r'$\Delta I$', fontsize = 14)
plt.xticks(fontsize = 14);plt.yticks(fontsize = 14)
plt.grid();plt.show()

#plt.figure(figsize = (7,5))
plt.plot(V_arr, 100*np.real(I_store-I_store2)/np.real(I_store + I_store2), linewidth = 2)

plt.xlabel('V', fontsize = 14);plt.ylabel('MR[%]', fontsize = 14)
plt.xticks(fontsize = 14);plt.yticks(fontsize = 14)
plt.grid();plt.show()

#%%
def n_up_app(V):
    return 1/N_sites*(N_sites/2*1 + 0.2*V)

def n_down_app(V):
    return 1/N_sites*(N_sites/2*1 - 0.15*V)

def n_up_diff(V):
    return 1/N_sites*0.0002*np.abs(V)

def n_down_diff(V):
    return 1/N_sites*0.0001*np.abs(V)


def n_up_pos_app(V):
    return n_up_app(V) + 0.5*n_up_diff(V)

def n_down_pos_app(V):
    return n_down_app(V) + 0.5*n_down_diff(V)

def n_up_neg_app(V):
    return n_down_app(V) - 0.5*n_down_diff(V)

def n_down_neg_app(V):
    return n_up_app(V) - 0.5*n_up_diff(V)

V_arr_app = np.arange(-10,10,2)

Occ_store_app = np.zeros([len(V_arr_app),N_sites,2,N_sites,2], dtype = 'complex128')
Occ_store_app2 = np.zeros([len(V_arr_app),N_sites,2,N_sites,2], dtype = 'complex128')

for i in range(0,N_sites):
    Occ_store_app[:,i,0,i,0] = n_down_pos_app(V_arr_app)
    Occ_store_app[:,i,1,i,1] = n_up_pos_app(V_arr_app)
    Occ_store_app2[:,i,0,i,0] = n_down_neg_app(V_arr_app)
    Occ_store_app2[:,i,1,i,1] = n_up_neg_app(V_arr_app)
#%%
    
    
def dFdU(E, mu, beta = 1):
    return beta/(8*np.cosh(beta/2*(E - mu))**2)

def IV_generate_approx(mu0, V_arr, e_arr_p, beta, Occ_store):
    """
    Computes the currents over a leads kept at chemical potential mu0 over a bias-voltage range specified by V_arr
    from the occupation and off-site capacitive arrays n_arr_sweep_res and UC_arr_sweep_res    
    n_arr_sweep_res[i], UC_arr_sweep_res[i] correspond to the voltage V_arr[i]
    
    """
    
    I_store_order0 = np.zeros([len(V_arr)], dtype = 'complex128')
    I_store_order1 = np.zeros([len(V_arr)], dtype = 'complex128')
    I_store_order2 = np.zeros([len(V_arr)], dtype = 'complex128')
    I_store_order3 = np.zeros([len(V_arr)], dtype = 'complex128')
    I_store_order4 = np.zeros([len(V_arr)], dtype = 'complex128')
    
    for j in range(0,len(V_arr)):
       # print(V_arr[j])
        
        FD_diff_arr = FD_dist(e_arr_p, mu0 + V_arr[j]/2, beta) - FD_dist(e_arr_p, mu0 - V_arr[j]/2, beta)
        print(j)
       # print(np.max(np.abs(FD_diff_arr)))
        Occ_j = Occ_store[j]
        Occ_res_j = np.reshape(Occ_j, [2*N_sites,2*N_sites])
        n_arr_vec_j = Occ_res_j[np.arange(0,2*N_sites),np.arange(0,2*N_sites)]
        
        n_arr_swap = np.zeros([2*N_sites], dtype = 'complex128')
        n_arr_swap[np.arange(0,2*N_sites,2)] = n_arr_vec_j[np.arange(1,2*N_sites,2)] - 0.
        n_arr_swap[np.arange(1,2*N_sites,2)] = n_arr_vec_j[np.arange(0,2*N_sites,2)] - 0.
        n_arr_swap_diag = np.diag(n_arr_swap)
        print(n_arr_swap_diag[:4,:4])
        
        n_zeros = np.zeros([2*N_sites], dtype = 'complex128')
       # n_arr_res_j = get_n_arr_res(Occ_j)
       # UC_arr_res_j = get_UC_arr_res(Occ_j)
        
        Transm_tot_order0 = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_tot_order1 = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_tot_order2 = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_tot_order3 = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_tot_order4 = np.zeros(len(e_arr_p), dtype = 'complex128')
        Transm_tot_nonint = np.zeros(len(e_arr_p), dtype = 'complex128')
        
        for i in range(len(e_arr_p)):
            G_ret_res = GF_ret(e_arr_p[i], n_zeros)#G_ret_store[i]#GF_ret(e_arr[i] + 0.0000001j,H0,Sigma_L_res + Sigma_R_res)
            G_ret = np.reshape(G_ret_res, [N_sites,2,N_sites,2])
            
            G_adv_res = np.transpose(np.conj(G_ret_res))
            G_adv = np.reshape(G_adv_res, [N_sites,2,N_sites,2])

            T_tot_nonint = np.trace(np.dot(Gamma_L_res, np.dot(G_ret_res, np.dot(Gamma_R_res, G_adv_res))))
            Transm_tot_nonint[i] = T_tot_nonint

            T_tot_arr0 = np.dot(Gamma_L_res, np.dot(G_ret_res, np.dot(n_arr_swap_diag, np.dot(G_ret_res, np.dot(Gamma_R_res, G_adv_res)))))
            T_tot_arr = T_tot_arr0*1 + 1*np.conj(np.transpose(T_tot_arr0))
            
           # print(np.trace(T_tot_arr))
            
            T_tot_arr0 = np.dot(Gamma_R_res, np.dot(G_ret_res, np.dot(n_arr_swap_diag, np.dot(G_ret_res, np.dot(Gamma_L_res, G_adv_res)))))
            T_tot_arr_order0 = T_tot_arr0 + 1*np.conj(np.transpose(T_tot_arr0))
            
           # print(np.trace(T_tot_arr))
           # print(i)
            
            T_tot_arr0 = np.dot(np.dot(Gamma_L_res, np.dot(G_ret_res, np.dot(n_arr_swap_diag, G_ret_res))), np.dot(Gamma_R_res, np.dot(G_adv_res, np.dot(n_arr_swap_diag, G_adv_res))))
            T_tot_arr10 = np.dot(Gamma_L_res, np.dot(G_ret_res, np.dot(n_arr_swap_diag, np.dot(G_ret_res, np.dot(n_arr_swap_diag, np.dot(G_ret_res, np.dot(Gamma_R_res, G_adv_res)))))))
            T_tot_arr1 = T_tot_arr10 + np.conj(np.transpose(T_tot_arr10))
            T_tot_arr_order1 = T_tot_arr0 + T_tot_arr1
            
            Gret_n = np.dot(G_ret_res, n_arr_swap_diag)
            Gadv_n = np.dot(G_adv_res, n_arr_swap_diag)
            Gret_n2 = np.dot(Gret_n, Gret_n)
            Gadv_n2 = np.dot(Gadv_n, Gadv_n)
            Gret_n3 = np.dot(Gret_n2, Gret_n)
            Gadv_n3 = np.dot(Gadv_n2, Gadv_n)
            Gret_n4 = np.dot(Gret_n3, Gret_n)
            Gadv_n4 = np.dot(Gadv_n3, Gadv_n)
            Gret_n5 = np.dot(Gret_n4, Gret_n)
            Gadv_n5 = np.dot(Gadv_n4, Gadv_n)
            
            T_tot_arr20 = np.dot(Gamma_L_res, np.dot(Gret_n3, np.dot(G_ret_res, np.dot(Gamma_R_res, G_adv_res))))
            T_tot_arr20 += np.dot(Gamma_L_res, np.dot(Gret_n2, np.dot(G_ret_res, np.dot(Gamma_R_res, np.dot(Gadv_n, G_adv_res)))))
            T_tot_arr20 += np.dot(Gamma_L_res, np.dot(Gret_n, np.dot(G_ret_res, np.dot(Gamma_R_res, np.dot(Gadv_n2, G_adv_res)))))
            T_tot_arr20 += np.dot(Gamma_L_res, np.dot(G_ret_res, np.dot(Gamma_R_res, np.dot(Gadv_n3, G_adv_res))))
            
            T_tot_arr_order2 = T_tot_arr20
            
            
            T_tot_arr30 = np.dot(Gamma_L_res, np.dot(Gret_n4, np.dot(G_ret_res, np.dot(Gamma_R_res, G_adv_res))))
            T_tot_arr30 += np.dot(Gamma_L_res, np.dot(Gret_n3, np.dot(G_ret_res, np.dot(Gamma_R_res, np.dot(Gadv_n, G_adv_res)))))
            T_tot_arr30 += np.dot(Gamma_L_res, np.dot(Gret_n2, np.dot(G_ret_res, np.dot(Gamma_R_res, np.dot(Gadv_n2, G_adv_res)))))
            T_tot_arr30 += np.dot(Gamma_L_res, np.dot(Gret_n, np.dot(G_ret_res, np.dot(Gamma_R_res, np.dot(Gadv_n3, G_adv_res)))))
            T_tot_arr30 += np.dot(Gamma_L_res, np.dot(G_ret_res, np.dot(Gamma_R_res, np.dot(Gadv_n4, G_adv_res))))
            
            T_tot_arr_order3 = T_tot_arr30
            
            T_tot_arr40 = np.dot(Gamma_L_res, np.dot(Gret_n5, np.dot(G_ret_res, np.dot(Gamma_R_res, G_adv_res))))
            T_tot_arr40 += np.dot(Gamma_L_res, np.dot(Gret_n4, np.dot(G_ret_res, np.dot(Gamma_R_res, np.dot(Gadv_n, G_adv_res)))))
            T_tot_arr40 += np.dot(Gamma_L_res, np.dot(Gret_n3, np.dot(G_ret_res, np.dot(Gamma_R_res, np.dot(Gadv_n2, G_adv_res)))))
            T_tot_arr40 += np.dot(Gamma_L_res, np.dot(Gret_n2, np.dot(G_ret_res, np.dot(Gamma_R_res, np.dot(Gadv_n3, G_adv_res)))))
            T_tot_arr40 += np.dot(Gamma_L_res, np.dot(Gret_n, np.dot(G_ret_res, np.dot(Gamma_R_res, np.dot(Gadv_n4, G_adv_res)))))
            T_tot_arr40 += np.dot(Gamma_L_res, np.dot(G_ret_res, np.dot(Gamma_R_res, np.dot(Gadv_n5, G_adv_res))))
            
            T_tot_arr_order4 = T_tot_arr40
            
            Transm_tot_order0[i] = np.trace(T_tot_arr_order0)
            Transm_tot_order1[i] = np.trace(T_tot_arr_order1)
            Transm_tot_order2[i] = np.trace(T_tot_arr_order2)
            Transm_tot_order3[i] = np.trace(T_tot_arr_order3)
            Transm_tot_order4[i] = np.trace(T_tot_arr_order4)

        #Spin_Pol = (Transm_arr_up + (Transm_arr_du - Transm_arr_ud) - Transm_arr_down)/Transm_tot
        
        I_store_order0[j] = np.abs(e_arr_p[1] - e_arr_p[0])*np.sum(Transm_tot_order0*FD_diff_arr)# + Transm_tot_nonint*(dFdU(e_arr_p, mu0 + V_arr[j]/2) - dFdU(e_arr_p, mu0 - V_arr[j]/2)))
        I_store_order1[j] = np.abs(e_arr_p[1] - e_arr_p[0])*np.sum(Transm_tot_order1*FD_diff_arr)
        I_store_order2[j] = np.abs(e_arr_p[1] - e_arr_p[0])*np.sum(Transm_tot_order2*FD_diff_arr)
        I_store_order3[j] = np.abs(e_arr_p[1] - e_arr_p[0])*np.sum(Transm_tot_order3*FD_diff_arr)
        I_store_order4[j] = np.abs(e_arr_p[1] - e_arr_p[0])*np.sum(Transm_tot_order4*FD_diff_arr)
    return I_store_order0, I_store_order1, I_store_order2, I_store_order3, I_store_order4, Transm_tot

I_store_order0, I_store_order1, I_store_order2, I_store_order3, I_store_order4, Transm_tot0 = IV_generate_approx(mu0, V_arr_app,e_arr, beta, Occ_store_app2)

I_store0, Transm_tot, Spin_Pol, Tup, Tdown, Tdu, Tud = IV_generate(mu0, V_arr_app,e_arr, beta, Occ_store_app2)

#%%
I_store_order02, I_store_order12, I_store_order22, I_store_order32, I_store_order42 = np.copy(I_store_order0), np.copy(I_store_order1), np.copy(I_store_order2), np.copy(I_store_order3), np.copy(I_store_order4)
I_store02 = np.copy(I_store0)

#%%

plt.plot(V_arr_app, I_store0 - I_store02)
plt.plot(V_arr_app, I_asymm_order0*U0 + I_asymm_order1*U0**2 +I_asymm_order2*U0**3 + I_asymm_order3*U0**4 + I_asymm_order4*U0**5)
plt.grid();plt.show()

#%%
I_asymm_order0 = np.copy(I_store_order0 - I_store_order02)
I_asymm_order1 = np.copy(I_store_order1 - I_store_order12)
I_asymm_order2 = np.copy(I_store_order2 - I_store_order22)
I_asymm_order3 = np.copy(I_store_order3 - I_store_order32)
I_asymm_order4 = np.copy(I_store_order4 - I_store_order42)




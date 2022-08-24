# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:40:53 2022

@author: janbr
"""

import numpy as np
import matplotlib.pyplot as plt
#%%

R_I = 1.4
R_M = 2
R_A = np.sqrt(7)

Phi1 = 60*np.pi/180
Phi2 = 19.1*np.pi/180

def Angle_site(N_hex, n_atom):
    del_phis = np.array([0,0,Phi2,Phi1-Phi2])
    phi_tot = N_hex*Phi1 + del_phis[n_atom]
    return phi_tot
        
def Spiral_map(n_atom):
    """
    Maps number of atom to helix-number
    0 => inner helix
    1 => middle helix
    2,3 => outer helix
    """
    Spiral_arr = np.array([0,1,2,2])
    return Spiral_arr[n_atom]

def Atom_pos(N_hex, n_atom, pitch):
    """
    Returns the position of atoms at the N_hex hexagon at the n_atom
    Input:
        N_hex : Either integer or an array of integers 
    Returns: 
        Pos : Where Pos[i,j] gives the i-th coordinate of the j-th atom
    """
    Spiral_N = Spiral_map(n_atom)
    Radius_dict = np.array([R_I,R_M,R_A])
    Radius_n = Radius_dict[Spiral_N]
    Angle_tot = Angle_site(N_hex, n_atom)
    Pos = np.array([Radius_n*np.cos(Angle_tot), Radius_n*np.sin(Angle_tot), pitch/(2*np.pi)*Angle_tot])
    return Pos
    
def Add_final_atoms(N_fin, pitch):
    """
    Adds positions of the 2 final atoms (since Atom_pos only generates the first 4 sites of the final hexagon)
    to the final hexagaon.
    """
    Pos1 = Atom_pos(N_fin + 1, 0, pitch); Pos2 = Atom_pos(N_fin + 1, 1, pitch)
    Pos_tot = np.zeros([3,2])
    Pos_tot[:,0] = Pos1; Pos_tot[:,1] = Pos2
    return Pos_tot

N_hex = 4
A_pos = (Atom_pos(np.linspace(0,N_hex-1,N_hex),0,1))
B_pos = (Atom_pos(np.linspace(0,N_hex-1,N_hex),1,1))
C_pos = (Atom_pos(np.linspace(0,N_hex-1,N_hex),2,1))
D_pos = (Atom_pos(np.linspace(0,N_hex-1,N_hex),3,1))
Fin_pos = (Add_final_atoms(N_hex-1,1))

plt.figure(figsize = (6,6))
plt.plot(A_pos[0,:], A_pos[1,:],'o')
plt.plot(B_pos[0,:], B_pos[1,:],'o')
plt.plot(C_pos[0,:], C_pos[1,:],'o')
plt.plot(D_pos[0,:], D_pos[1,:],'o')
plt.plot(Fin_pos[0,:], Fin_pos[1,:],'o')

plt.xlim(-R_A-R_I,R_A+R_I)
plt.ylim(-R_A-R_I,R_A+R_I)
plt.grid()
plt.show()

#%%

def Angle_mult_site(Ind_arr):
    """
    Gives the angles of the indices
    Input:
        Ind_arr : Indices, where Ind_arr[0] stores the hexagons' indices, while Ind_arr[1] stores the atoms' indices
        Beyond the 1st argument/index, Ind_arr can have any shape
    Returns:
        Phi_tot : Array storing the angles of the atoms in Ind_arr. phi_tot has the same shape as Ind[0] and Ind[1]. 
        Indices in phi_tot[:,...] correspond to indices in Ind[0,:,..]
    """
    Ind_Hex = Ind_arr[0]
    Ind_Atom = Ind_arr[1]
    del_phis = np.array([0,0,Phi2,Phi1-Phi2])
    phi_tot = Ind_Hex*Phi1 + del_phis[Ind_Atom]
    return phi_tot

def Spiral_mult_map(Ind_arr):
    """
    Gives the spiral/helix on which the atoms of Ind_arr lie
    Input:
        Ind_arr : Indices, where Ind_arr[0] stores the hexagons' indices, while Ind_arr[1] stores the atoms' indices
        Beyond the 1st argument/index, Ind_arr can have any shape
    Returns:
        Spiral_arr : Helix on which the atoms are on. 0 corresponds to the inner helix, 1 <=> middle helix, 2 <=> outer helix
        Spiral_arr has the same shape as Ind_arr, while Spiral_arr[i,j,...] gives the helix on which the atom corresponding 
        to Ind_arr[i,j,...] lies on
    """
    Ind_Atom = Ind_arr[1]
    Atom_map = np.array([0,1,2,2])
    Spiral_arr = Atom_map[Ind_Atom]
    return Spiral_arr

def Atom_mult_pos(Ind_arr, pitch):
    """
    Gives the positions of atoms whose indices are stored in Ind_arr
    Input:
        Ind_arr : Indices, where Ind_arr[0] stores the hexagons' indices, while Ind_arr[1] stores the atoms' indices
        Beyond the 1st argument/index, Ind_arr can have any shape
    Returns:
        Pos : Storing the positions of the atoms. Pos[0], Pos[1], Pos[2] give the x,y,z-coordinates (respectively)
        of all the atoms in Ind_arr. Pos[x_i,i,j,...] gives the i-th coordinate of the atom corresponding to 
        Ind_arr[i,j,...]
    """
    Ind_Hex = Ind_arr[0]
    Ind_Atom = Ind_arr[1]
    Angles = Angle_mult_site(Ind_arr)
    Spiral_indices = Spiral_mult_map(Ind_arr)
    Radius_dict = np.array([R_I,R_M,R_A])
    Radius_vals = Radius_dict[Spiral_indices]
    Pos = np.array([Radius_vals*np.cos(Angles), Radius_vals*np.sin(Angles), pitch/(2*np.pi)*Angles])
    return Pos

def Atom_site_vecs(Ind_arr):
    """
    Gives the set of base/normal-vectors at sites stored in Ind_arr
    Input:
        Ind_arr : Indices, where Ind_arr[0] stores the hexagons' indices, while Ind_arr[1] stores the atoms' indices
        Beyond the 1st argument/index, Ind_arr can have any shape
    Returns:
        base_vecs : where base_vecs[i,j,k,...] gives the j-th coordinate of the i-th normal/base-vector of the site
        corresponding to the site of indices [k,...] in the Ind_arr
    """
    Ind_Hex = Ind_arr[0]
    Ind_Atom = Ind_arr[1]
    Angles = Angle_mult_site(Ind_arr)
    nx = np.array([np.cos(Angles), np.sin(Angles),np.zeros([len(Angles)])])
    ny = np.array([-np.sin(Angles), np.cos(Angles),np.zeros([len(Angles)])])
    nz = np.array([np.zeros([len(Angles)]), np.zeros([len(Angles)]), np.ones([len(Angles)])])
    base_vecs = np.array([nx,ny,nz])
    return base_vecs



#%%


def index_convert(N_hex):
    """
    Input:
        N_hex : Number of hexagons of which the helix is constructed
    Returns:
        ivals1_res : i_vals2_res[i,j] gives the converted index corresponding to the j-th atom in the i-th hexagon
    """
    i_vals1 = np.linspace(0,N_hex*4-1,N_hex*4)
    i_vals1_res = np.reshape(i_vals1, [N_hex,4])
    return i_vals1_res.astype(int)

def rev_index_convert(N_hex):
    """ 
    Input:
        N_hex : Number of hexagons of which the helix is constructed
    Returns:
        Ind_arr : Ind_arr[:,i] gives the original indices (hexagon-index & atom-index) of the converted index i
    """
    Ind = np.linspace(0,4*N_hex-1,4*N_hex).astype(int)
    ind1 = np.floor(Ind/4)
    ind2 = Ind - ind1*4
    Ind_arr = np.array([ind1,ind2])
    return Ind_arr.astype(int)



def generate_connections(N_hex):
    """
    Generates links between atoms for overlapping electrons
    Input:
        N_hex : Number of hexagons of which the helix is constructed
    Returns:
        Conns_tot : Conns_tot[:,i] gives the (converted) indices of the i-th link
    """
    Indices_arr = index_convert(N_hex)
    Conns_01 = np.array([Indices_arr[:,0], Indices_arr[:,1]])
    Conns_12 = np.array([Indices_arr[:,1], Indices_arr[:,2]])
    Conns_23 = np.array([Indices_arr[:,2], Indices_arr[:,3]])
    Conns_inter_31 = np.array([Indices_arr[:-1,3], Indices_arr[1:,1]])
    Conns_inter_00 = np.array([Indices_arr[:-1,0], Indices_arr[1:,0]])

    Conns_tot = np.copy(Conns_01)
    Conns_tot = np.append(Conns_tot, Conns_12, axis = 1)
    Conns_tot = np.append(Conns_tot, Conns_23, axis = 1)
    Conns_tot = np.append(Conns_tot, Conns_inter_31, axis = 1)
    Conns_tot = np.append(Conns_tot, Conns_inter_00, axis = 1)
    return Conns_tot.astype(int)
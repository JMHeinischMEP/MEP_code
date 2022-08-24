# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:09:58 2022

@author: janbr
"""

import numpy as np
import matplotlib.pyplot as plt

#%%

def get_e_comp(e_arr):
    len0 = 5
    e_len = len(e_arr)
    e_init, e_fin = e_arr[0], e_arr[-1]
   # print(e_len)
    len_comp = int(len0*np.floor(e_len/len0))
   # print(len_comp)
    e_comp = np.linspace(e_init, e_fin, len_comp)
   # print(e_comp[:len0])
    return e_comp
    
get_e_comp(np.linspace(-15,15,30000))

def slice_e_comp(e_arr):
    len0 = 5
    len_partial = 15000
    len_e0 = len(e_arr)
    e_init, e_fin = e_arr[0], e_arr[-1]
    e_list = []
    
    #Checking whether e_arr can be cut into slices with no remainder
    Smooth_bool = (int(np.floor(len_e0/len_partial)) == len_e0/len_partial)

    n_slice = int(np.floor(len_e0/len_partial)) + 1

    len_tot0 = len_partial + (len_partial - 1)*np.max([(n_slice - 2),0])
    len_fin0 = len_e0 - len_tot0
    print(len_fin0)
    len_fin = np.max([len0*int((len_fin0 + 0)/len0) - 1,len0 - 1])
    len_tot = len_tot0 + len_fin
    
    e_slice = np.linspace(e_init, e_fin, len_tot)
    print(len_tot0)
    print(len_fin)
    index0, index1 = 0, len_partial
    for i in range(0,n_slice):
        e_slice_i = e_slice[index0:index1]
        index0 = index1 - 1
        index1 += len_partial - 1
        print(len(e_slice_i))
        e_list.append(e_slice_i)
    return e_list


def slice_e_comp(e_arr):
    len0 = 5
    len_partial = 15000
    len_e0 = len(e_arr)
    e_init, e_fin = e_arr[0], e_arr[-1]
    e_list = []
    
    #Checking whether e_arr can be cut into slices with no remainder
    Smooth_bool = (int(np.floor(len_e0/len_partial)) == len_e0/len_partial)
    print(Smooth_bool)
    if Smooth_bool:
        n_slice = int(np.floor(len_e0/len_partial))
        len_tot0 = len_partial + (len_partial - 1)*np.max([(n_slice - 1),0])
        print(n_slice)
        print(len_tot0)
        len_tot = np.copy(len_tot0)
    if not(Smooth_bool):
        n_slice = int(np.floor(len_e0/len_partial)) 
        if n_slice > 0:
            len_tot0 = len_partial + (len_partial - 1)*np.max([(n_slice - 1),0])
            print(n_slice)
            print(len_tot0)
            delta_len = len_e0 - len_tot0
            len_fin = np.max([len0*int(np.floor(delta_len/len0)) -1 ,0])
            print(len_fin)
            if len_fin > 0:
                n_slice += 1
        if n_slice == 0:
            len_tot0 = len_partial
            len_fin = 0
            n_slice = 1
            
        len_tot = len_tot0 + len_fin
    e_slice = np.linspace(e_init, e_fin, len_tot)
    index0, index1 = 0, len_partial
    for i in range(0,n_slice):
        e_slice_i = e_slice[index0:index1]
        index0 = index1 - 1
        index1 += len_partial - 1
        print(len(e_slice_i))
        e_list.append(e_slice_i)
    return e_list
    
    
e_list = slice_e_comp(np.linspace(-15,15,60000))   



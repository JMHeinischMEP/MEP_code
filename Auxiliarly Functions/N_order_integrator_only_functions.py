# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 12:11:25 2022

@author: janbr
"""
#%%

import scipy.special as scs
import numpy as np
import matplotlib.pyplot as plt 
#%%
def integrate_n(Arr, n_order, h):
    """
    Computes the integral of array, performed over the last index in an n-th order approximation
    Values over which is integrated should be evenly spaced in the integration axis
    Input:
        Arr : (shape = [arbitrary-shape]) Array of arbitrary shape. The integration is performed over the last axis 
        n_order : (int, n_order >= 1) Order for which the integration is performed (n_order = 1)
        h : (float) spacing between the points on the integration-axis
    Returns:
        Integral : (shape = Arr.shape[:-1]) Array containing the values of the integrals performed on the input-array Arr
    """
    Arr_shape0 = Arr.shape[:-1]; Arr_shapefin = Arr.shape[-1]
    n_sections = int(Arr_shapefin/(2*n_order - 1))
    Arr_newshape = Arr_shape0 + (n_sections, 2*n_order - 1)
    Arr_res = np.reshape(Arr, Arr_newshape)
    
    Coeff_arr = np.zeros([2*n_order - 1, 2*n_order - 1], dtype = 'float64')
    
    factor_arr =1/scs.factorial(np.arange(0,2*n_order - 1));
    factor_arr_adv = 1/scs.factorial(np.arange(1,2*n_order))
    
    pow_arr = np.arange(0,2*n_order - 1)
    pow_arr_adv = np.arange(1,2*n_order)
    
    Coeff_arr[n_order - 1,0] = 1
    for i in range(1,n_order):
        xval_pos = i*h; xval_neg = -i*h
        Coeff_arr[n_order - 1 - i] = factor_arr*xval_pos**pow_arr; Coeff_arr[n_order - 1 + i] = factor_arr*xval_neg**pow_arr

    Coeff_arr_inv = np.linalg.inv(Coeff_arr)
    Taylor_coeffs = (np.tensordot(Arr_res, Coeff_arr_inv, axes = ([-1], [-1])))

    xmax = h*(n_order - 1 + 1/2); xmin = -h*(n_order - 1 + 1/2)
    xmax_fin = h*(n_order - 1 + 1/2*0); xmin_init = -h*(n_order - 1 + 1/2*0)

    vals_arr = factor_arr_adv*xmax**pow_arr_adv - factor_arr_adv*xmin**pow_arr_adv
    vals_arr_init = factor_arr_adv*xmax**pow_arr_adv - factor_arr_adv*xmin_init**pow_arr_adv
    vals_arr_fin = factor_arr_adv*xmax_fin**pow_arr_adv - factor_arr_adv*xmin**pow_arr_adv

    Partial_sums = np.tensordot(Taylor_coeffs, vals_arr, axes = ([-1],[0]))
    
    init_ind = (slice(0,None),)*len(Arr_shape0) + (-1,)
    fin_ind = (slice(0,None),)*len(Arr_shape0) + (0,)
    Partial_sums[fin_ind]  = np.tensordot(Taylor_coeffs[fin_ind], vals_arr_fin, axes = ([-1],[0]))
    Partial_sums[init_ind]  = np.tensordot(Taylor_coeffs[init_ind], vals_arr_init, axes = ([-1],[0]))
   
    Integral = np.sum(Partial_sums, axis = -1)
    
    print(Integral)
    return Integral
    
def integrate_n_flipaxes(Arr_f, n_order, h):
    """
    Computes the integral of array, performed over the first index in an n-th order approximation
    Values over which is integrated should be evenly spaced in the integration axis
    Input:
        Arr : (shape = [arbitrary-shape]) Array of arbitrary shape. The integration is performed over the first axis 
        n_order : (int, n_order >= 1) Order for which the integration is performed (n_order = 1)
        h : (float) spacing between the points on the integration-axis
    Returns:
        Integral : (shape = Arr.shape[1:]) Array containing the values of the integrals performed on the input-array Arr
    """
    Arr = np.moveaxis(Arr_f, 0,-1)
  #  print(Arr.shape)
    Arr_shape0 = Arr.shape[:-1]; Arr_shapefin = Arr.shape[-1]
    n_sections = int(Arr_shapefin/(2*n_order - 1))
    Arr_newshape = Arr_shape0 + (n_sections, 2*n_order - 1)
    Arr_res = np.reshape(Arr, Arr_newshape)
    
    Coeff_arr = np.zeros([2*n_order - 1, 2*n_order - 1], dtype = 'float64')
    
    factor_arr =1/scs.factorial(np.arange(0,2*n_order - 1));
    factor_arr_adv = 1/scs.factorial(np.arange(1,2*n_order))
    
    pow_arr = np.arange(0,2*n_order - 1)
    pow_arr_adv = np.arange(1,2*n_order)
    
    Coeff_arr[n_order - 1,0] = 1
    for i in range(1,n_order):
        xval_pos = i*h; xval_neg = -i*h
        Coeff_arr[n_order - 1 - i] = factor_arr*xval_pos**pow_arr; Coeff_arr[n_order - 1 + i] = factor_arr*xval_neg**pow_arr

    Coeff_arr_inv = np.linalg.inv(Coeff_arr)
    Taylor_coeffs = (np.tensordot(Arr_res, Coeff_arr_inv, axes = ([-1], [-1])))

    xmax = h*(n_order - 1 + 1/2); xmin = -h*(n_order - 1 + 1/2)
    xmax_fin = h*(n_order - 1 + 1/2*0); xmin_init = -h*(n_order - 1 + 1/2*0)

    vals_arr = factor_arr_adv*xmax**pow_arr_adv - factor_arr_adv*xmin**pow_arr_adv
    vals_arr_init = factor_arr_adv*xmax**pow_arr_adv - factor_arr_adv*xmin_init**pow_arr_adv
    vals_arr_fin = factor_arr_adv*xmax_fin**pow_arr_adv - factor_arr_adv*xmin**pow_arr_adv

    Partial_sums = np.tensordot(Taylor_coeffs, vals_arr, axes = ([-1],[0]))
    
    init_ind = (slice(0,None),)*len(Arr_shape0) + (-1,)
    fin_ind = (slice(0,None),)*len(Arr_shape0) + (0,)
    Partial_sums[fin_ind]  = np.tensordot(Taylor_coeffs[fin_ind], vals_arr_fin, axes = ([-1],[0]))
    Partial_sums[init_ind]  = np.tensordot(Taylor_coeffs[init_ind], vals_arr_init, axes = ([-1],[0]))
   
    Integral = np.sum(Partial_sums, axis = -1)
    #print(Integral.shape)
   # Integral_f = np.swapaxes(Integral, 0,-1)
   # print(Integral_f.shape)
    #print(Integral)
    return Integral

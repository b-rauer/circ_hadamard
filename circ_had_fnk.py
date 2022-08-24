#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:23:22 2022

@author: bernhard
"""

import numpy as np
from scipy.linalg import hadamard

def get_hadamard(N_mode):
    '''
    Returns regular square hadamard patterns

    Parameters
    ----------
    N_mode : int
        Number of modes. Needs to be power of 2.

    Returns
    -------
    had_pattern : uint8
        Patterns arranged in 2D.
    had_matrix : float64
        Hadamard matrix.

    '''
    had_matrix = hadamard(N_mode**2)
    
    had_pattern = np.zeros((N_mode**2,N_mode,N_mode))
    for idx_mode in range(N_mode**2):
        had_pattern[idx_mode,:,:] = np.reshape( had_matrix[idx_mode,:], (N_mode, N_mode) )
       
    had_pattern = had_pattern.clip(min=0).astype('uint8')
    had_matrix = had_matrix / N_mode**2.
    
    return had_pattern, had_matrix

def get_circ_hadamard(N_mode):
    '''
    Returns circular Hadamard patterns on a square region.

    Parameters
    ----------
    N_mode : int
        Number of modes. Needs to be power of 2. Only 16 and 32 work here.

    Returns
    -------
    had_pattern : uint8
        Patterns arranged in 2D.
    had_matrix : float64
        Hadamard matrix.

    '''
    if N_mode < 16:
        
        had_pattern, had_matrix_circ = get_hadamard(N_mode)
        
    else:
        
        delta_r = 0.53
        width = int(N_mode + N_mode/8)
        
        if N_mode == 16:
            add_px_flag = 0
            add_px = []
        elif N_mode == 32:
            add_px_flag = 1
            add_px = np.zeros((2,4))
            add_px[0,:] = [0, 13, 35, 22]
            add_px[1,:] = [13, 35, 22, 0]
           
        x = np.linspace(-width/2,width/2,width)    
        base_mask = np.zeros((width,width))
    
        for idx1 in range(width):
            for idx2 in range(width):
                if np.sqrt(x[idx1]**2+x[idx2]**2) < (width/2 + delta_r):
                    base_mask[idx1,idx2] = 1
              
        if add_px_flag:       
            for idx in range(add_px.shape[1]):
                base_mask[int(add_px[0,idx]),int(add_px[1,idx])] = 1
    
        pixel_idx = np.where(base_mask == 1)
    
        had_matrix = hadamard(N_mode**2);
#        had_matrix = had_matrix.clip(min=0);
#        had_matrix = np.uint8(had_matrix);
    
#        had_pattern = np.zeros((N_mode**2,width,width),dtype=np.uint8)
        had_pattern = np.zeros((N_mode**2,width,width))
        for idx_mode in range(N_mode**2):
            for idx_px in range(pixel_idx[0].shape[0]):
                had_pattern[idx_mode,pixel_idx[0][idx_px],pixel_idx[1][idx_px]] = had_matrix[idx_mode,idx_px]
                
        had_matrix_circ = np.zeros((N_mode**2,width**2))
        for idx_mode in range(N_mode**2):
            had_matrix_circ[idx_mode,:] = np.reshape(had_pattern[idx_mode,:,:], (width**2,))
        
        had_pattern = had_pattern.clip(min=0).astype('uint8')
        had_matrix_circ = had_matrix_circ / N_mode**2.
        
    return had_pattern, had_matrix_circ

def get_double_circ_hadamard(N_mode):
    '''
    Returns a doubel set of circular Hadamard patterns on a square region. The two sets each cover half of the circle. They are defined randomly.

    Parameters
    ----------
    N_mode : int
        Number of modes. Needs to be power of 2. Only 8, 16 and 32 work here.

    Returns
    -------
    had_pattern : uint8
        Patterns arranged in 2D.
    had_matrix : float64
        Hadamard matrix.

    '''
    if N_mode == 8:
        width = 12
        delta_r = 0.9
        switch_val = 1
        switch_px = np.zeros((2,8))
        switch_px[0,:] = [1, 1, 10, 10, 9, 11, 2, 0]
        switch_px[1,:] = [1, 10, 1, 10, 0, 9, 11, 2]
    elif N_mode == 16:
        width = 26
        delta_r = 0.4
        switch_val = 0
        switch_px = np.zeros((2,12))
        switch_px[0,:] = [0, 10, 25, 10, 15, 0, 15, 25, 7, 24, 18, 1]
        switch_px[1,:] = [10, 0, 10, 25, 0, 15, 25, 15, 1, 7, 24, 18]
    elif N_mode == 32:
        width = 52
        delta_r = 0.1
        switch_val = 0
        switch_px = np.zeros((2,8))
        switch_px[0,:] = [1, 1, 50, 50, 19, 19, 32, 32]
        switch_px[1,:] = [19, 32, 19, 32, 1, 50, 1, 50]
           
    x = np.linspace(-width/2,width/2,width)    
    base_mask = np.zeros((width,width))
    
    for idx1 in range(width):
        for idx2 in range(width):
            if np.sqrt(x[idx1]**2+x[idx2]**2) < (width/2 + delta_r):
                base_mask[idx1,idx2] = 1
    
    for idx in range(switch_px.shape[1]):
        base_mask[int(switch_px[0,idx]),int(switch_px[1,idx])] = switch_val

    pixel_idx = np.where(base_mask == 1)
   
    pixel_order = np.random.permutation(2*N_mode**2)
   
    had_matrix = hadamard(N_mode**2);

    had_pattern = np.zeros((2*N_mode**2,width,width))
    for idx_mode in range(N_mode**2):
        for idx_px in range(N_mode**2):
            
            pixel_idx_set1 = [pixel_idx[0][pixel_order[idx_px]], pixel_idx[1][pixel_order[idx_px]]]
            had_pattern[idx_mode,pixel_idx_set1[0],pixel_idx_set1[1]] = had_matrix[idx_mode,idx_px]
            
            pixel_idx_set2 = [pixel_idx[0][pixel_order[N_mode**2 + idx_px]], pixel_idx[1][pixel_order[N_mode**2 + idx_px]]]
            had_pattern[N_mode**2 + idx_mode,pixel_idx_set2[0],pixel_idx_set2[1]] = had_matrix[idx_mode,idx_px]
                
    had_matrix_circ = np.zeros((2*N_mode**2,width**2))
    for idx_mode in range(2*N_mode**2):
        had_matrix_circ[idx_mode,:] = np.reshape(had_pattern[idx_mode,:,:], (width**2,))
        
    had_pattern = had_pattern.clip(min=0).astype('uint8')
    had_matrix_circ = had_matrix_circ / N_mode**2.
        
    return had_pattern, had_matrix_circ


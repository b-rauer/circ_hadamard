#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:28:03 2022

@author: bernhard
"""

import matplotlib.pyplot as plt
import random
import circ_had_fnk

N_mode = 32

had_pattern, had_matrix_circ = circ_had_fnk.get_circ_hadamard(N_mode)

plot_mode_idx = random.randint(0, N_mode**2-1)

plt.imshow(had_pattern[plot_mode_idx,:,:])
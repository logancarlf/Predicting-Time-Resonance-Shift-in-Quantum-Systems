# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:24:50 2022

@author: PSClo
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import os
import pandas as pd
import shutil
#%%
#Only run once you rerun main and test.py

def data_move():
    
    #changing path source so others can use
    source = os.path.dirname(os.path.realpath('stats.py'))
    data_dest = r'C:\Imperial\Bsc\data\omega_dist_data'
    #data_source = source
    data_source = r'C:\Users\PSClo\OneDrive\Miscallaneous\Documents\Imperial_physics\Bsc_project\Predicting-Time-Resonance-Shift-in-Quantum-Systems\data1.txt'
    shutil.copy(data_source, data_dest)
    os.remove(r'C:\Users\PSClo\OneDrive\Miscallaneous\Documents\Imperial_physics\Bsc_project\Predicting-Time-Resonance-Shift-in-Quantum-Systems\data1.txt')
    
    return


# =============================================================================
# def lorentzian(x, a, x0):
#     return a / ((x-x0)**2 + a**2) / np.pi
# =============================================================================

def lorentzian(x,a, x_0, gamma):
    
    return (a * (1/np.pi) * (0.5*gamma))/((x-x_0)**2 + (0.5*gamma)**2)

def lorentz(x, a, x_0, gamma):
    return a/(1 + ((x-x_0)/gamma)**2)

#%%
df_omegadist = pd.read_csv('C:\Imperial\Bsc\data\omega_dist_data\data1.txt', sep = ' ', header = None)
pert_freq = df_omegadist[0]
max_exc_p = df_omegadist[1]


plt.plot(pert_freq, max_exc_p)

x0_guess = pert_freq[max_exc_p.idxmax()]
gamma_guess = 0.1*pert_freq.max()
init_guess = [np.max(max_exc_p), x0_guess, gamma_guess]

#popt, ier = leastsq(lorentzian, 1, args = (pert_freq, max_exc_p))
popt, pcov = curve_fit(lorentzian, pert_freq, max_exc_p, p0 = init_guess)
#plt.plot(pert_freq, lorentzian(pert_freq, *popt))

a_fit = popt[0]
x0_fit = popt[1]
gamma_fit = popt[2]

plt.plot(pert_freq, lorentzian(pert_freq, a_fit, x0_fit, gamma_fit))
#%%





#gamma specifies width 
#x_0 is location of peak 


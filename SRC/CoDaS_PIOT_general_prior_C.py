#!/usr/bin/env python
"""
Probabilistic Inverse Optimal Transport
Wei-Ting Chiu
Created on 05/06/2021
Modified MC 07/22/2021
Modified MC, new burn in for each Markov chain, 08/16/2021
"""
import Sinkhorn

import numpy as np
import pandas as pd
import random
import scipy
from scipy.stats import dirichlet
import matplotlib.pylab as plt
import torch
from scipy.stats import norm
# import ot
# import ot.plot
import math
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def scaleRowGaussian(K, r):
    r = torch.exp(r)
    r = r.reshape(-1, 1)
    K = K * r
    
    return K
    
def scaleColGaussian(K, c):
    c = torch.exp(c)
    K = K * c
    
    return K

def funScale(x, sigma, delta):
    return sigma * np.power(x, 3) + delta

def calculateStd(x, row_or_col):
    if row_or_col == 'row':
        axis = 1
    elif row_or_col == 'col':
        axis = 0
    return x.sum(axis = axis, dtype = torch.float)

def compute_ratio(K, K_prime, std, shift, row_or_col):
    ratio = 0.0

    x = calculateStd(K, row_or_col)
    x_prime = calculateStd(K_prime, row_or_col)

    for i in range(len(x)):
        ratio += np.log((norm.pdf( np.log(x[i])-np.log(x_prime[i]), \
                        scale = funScale(x_prime[i], std, shift))))\
                -np.log((norm.pdf( np.log(x_prime[i])-np.log(x[i]), \
                        scale = funScale(x[i], std, shift) )))
    
    return np.exp(ratio)
    

def compute_ratio_asym(x, x_prime, rho, row_or_col, idx):
    # Compute ratio for asymetric proposal distribution [1/rho, rho]
    ratio = 0.0

    prop = calculateStd(x, row_or_col)
    prop_prime = calculateStd(x_prime, row_or_col)

    ratio = np.log(prop[idx])-np.log(prop_prime[idx])
    
    return np.exp(ratio)

def monteCarloOneStep(K_old, prob_old, alphas, lam, mean, std, shift):

    acceptFlag = False
    prob_new = 1.0

    nr, nc = len(K_old), len(K_old[0])

    # Proposal moves for symmetric cost
    fac = np.random.normal(loc = 0.0, scale = funScale(torch.ones(nr+nc), std, shift))
    fac[nr:] = - fac[0:nr]
    fac = torch.tensor(fac)

    K_prime = scaleRowGaussian(K = K_old.clone(), r = fac[0:nr])
    K_new = scaleColGaussian(K = K_prime.clone(), c = fac[nr:])

    if torch.max(K_new) >= 1.0:
        # Reject if max(K') > 1, since C becomes negative which is not allowed
        return K_old.clone(), prob_old, acceptFlag
    else:

        C_new = -np.log(K_new.clone())/lam
        #C_new = C_new/C_new.sum()
        # C divided by sum for numerical stability
        prob_new = prob_new * dirichlet.pdf((C_new/C_new.sum()).reshape((-1,)), alphas)
        
        hastings_ratio = 1.0

        ratio = hastings_ratio * prob_new / prob_old 
        rv = random.random()
        if ratio > rv:
            acceptFlag = True
            return K_new.clone(), prob_new, acceptFlag
        else:
            return K_old.clone(), prob_old, acceptFlag


def monteCarloOneSweep(K, prob, alphas, lam, mean, std, shift, acceptCounter):
    

    K, prob, acceptFlag = monteCarloOneStep(K, prob, alphas, lam, mean, std, shift)
    if acceptFlag:
        acceptCounter = acceptCounter + 1
            
    return K.clone(), prob, acceptCounter

    
    
def monteCarlo(T, alphas, lam, mean, std, shift, max_iter, burnInFlag, num_lag, C_old = None, prob_old = None):
    
    '''
    Initialization
    '''
    data_C = []
    acceptCounter = 0
    lowerBound = 1e-5

    if C_old is None:
        K_old = T.clone()
        C_old = -np.log(K_old)/lam
        #C_old = C_old / C_old.sum()
    else:
        K_old = np.exp(-lam*C_old)
    n_row, n_col = len(K_old), len(K_old[0])


    if prob_old is None:
        prob_old = 1.0
        C_tmp = C_old.clone()/C_old.sum()
        prob_old = prob_old * dirichlet.pdf( C_tmp.reshape((-1,)), alphas)

    # print('Prob. old: %1.2e' % prob_old)
    
    for i in tqdm(range(max_iter)):       
        K_new, prob_new, acceptCounter = monteCarloOneSweep(K_old.clone(), prob_old, alphas, lam, mean, std, shift, acceptCounter)
        if not burnInFlag:
            if i % num_lag == 0 and not burnInFlag:
                data_C.append(-np.log(K_new)/lam)
        else:
            data_C.append(-np.log(K_new)/lam)
        K_old, prob_old = K_new.clone(), prob_new

    # print('Prob. new: %1.2e' % prob_new)
            
    return -np.log(K_old)/lam, prob_old, data_C, float(acceptCounter)/float(max_iter)
        

def runs(T, alphas, lam, mean, std, shift, num_burn_in, num_sampling, num_lag, C = None):
    nr, nc = len(T), len(T[0])
    data_C = []

    print("Burn in steps:")
    burnInFlag = True
    C_burn_in, prob_burn_in, data_C_burn_in, acceptRatio = monteCarlo(T, alphas, lam, mean, std, shift, num_burn_in, burnInFlag, num_lag)

    print("Burn in acceptance Ratio: %1.2f" % acceptRatio)

    print("Sampling steps:")
    burnInFlag = False

    _, _, data_C_tmp, acceptRatio = monteCarlo(T, alphas, lam, mean, std, shift, int(num_sampling), burnInFlag, num_lag, C_burn_in, prob_burn_in)
    data_C = data_C + data_C_tmp

    print("Sampling acceptance Ratio: %1.2f" % acceptRatio)
    print('Size of data K: %i' % len(data_C))

    return C, data_C, data_C_burn_in, acceptRatio



if __name__ == '__main__':
    pass

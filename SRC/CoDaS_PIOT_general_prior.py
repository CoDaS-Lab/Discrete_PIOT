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
import math
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def sim_SK(op, nr, nc, reg = 1.0):
    '''
    Generate a random K matrix with distribution defined in op and 
    perform Sinkhorn algorithm to get the OP plan T.
    '''

    print('--', op, '--')

    if op[0] == 'Uniform':
        K, r, c = CoDaS_Sinkhorn.genM(nr, nc)
        K = K / K.sum()

    elif op[0] == 'Dirichlet':
        alpha = op[1]
        K, r, c = CoDaS_Sinkhorn.genM_Dirichlet(nr, nc, alpha)

    elif op[0] == 'Gaussian':
        mean = op[1]
        std = op[2]
        K, r, c = CoDaS_Sinkhorn.genM_Normal(nr, nc, mean, std)
        K = K / K.sum()


    ''''''
    # r = r / r.sum()
    # c = c / c.sum()   
    
        
    stopThr = 1e-9
    numItermax = 10000
    T = CoDaS_Sinkhorn.sinkhorn_torch(mat = K.clone(), 
                                            row_sum = r, 
                                            col_sum = c, 
                                            epsilon = stopThr, 
                                            max_iter = numItermax)
    
    print("K")
    print(K, '\n')
    print("T")
    print(T, '\n')
    
    return K, T, r, c

def relative_error(A, B):
    '''
    Calculate relative error between A and B.
    error = sqrt( sum_{ij} (A_{ij} - B_{ij})^2 ) / sqrt( sum_{ij} B_{ij}^2 )
    '''
    err = np.sqrt( np.square(np.subtract(A, B)).sum() ) / np.sqrt( np.square(B).sum() )
    return err

def sinkhorn(m, n, reg, stopThr, numItermax):
    t1 = time.time()
    alpha = 1.0
    K, r, c = CoDaS_Sinkhorn.genM_Dirichlet(m, n, alpha)
    K = K / K.sum()
    r = r / r.sum()
    c = c / c.sum()
    mean = K.mean()
    
    T = CoDaS_Sinkhorn.sinkhorn_torch(mat=K,
                                    row_sum = r, 
                                    col_sum = c, 
                                    epsilon=stopThr, 
                                    max_iter=numItermax)
    
    t2 = time.time()
    return T

def scaleRowGaussian(K, mean, std, idx):
    rv = torch.normal(mean = mean, std = std)
    rv = torch.exp(rv)
    r = torch.ones(len(K))
    r[idx] = rv
    r = r.reshape(-1, 1)
    K = K * r
    
    return K

def scaleColGaussian(K, mean, std, idx):
    rv = torch.normal(mean = mean, std = std)
    rv = torch.exp(rv)
    c = torch.ones(len(K[0]))
    c[idx] = rv
    K = K * c
    
    return K


def funScale(x, sigma, delta):
    return sigma * np.power(x, 3) + delta
    #return torch.tensor(delta)
    #return sigma*np.power(x, -1)+delta

def calculateStd(x, row_or_col):
    if row_or_col == 'row':
        axis = 1
    elif row_or_col == 'col':
        axis = 0
    return x.sum(axis = axis, dtype = torch.float)

def compute_ratio(x, x_prime, std, shift, row_or_col, idx):
    ratio = 0.0

    prop = calculateStd(x, row_or_col)
    prop_prime = calculateStd(x_prime, row_or_col)

    if row_or_col == 'row':
        n = len(x)
    elif row_or_col == 'col':
        n = len(x[0])


    ratio = np.log((norm.pdf( np.log(prop[idx])-np.log(prop_prime[idx]), \
                        scale = funScale(prop_prime[idx], std, shift))))\
        -np.log((norm.pdf( np.log(prop_prime[idx])-np.log(prop[idx]), \
                        scale = funScale(prop[idx], std, shift) )))
    
    return np.exp(ratio)


def monteCarloOneStep_changingStd(K_old, prob_old, alphas, lam, mean, std, shift, row_or_col, idx):

    acceptFlag = False
    prob_new = 1.0
    n_col = len(K_old[0])

    if row_or_col == 'row':
        rc = calculateStd(K_old, row_or_col)
        K_prime = scaleRowGaussian(K = K_old.clone(), mean = mean, std = funScale(rc[idx], std, shift), idx = idx)
    
    elif row_or_col == 'col':
        rc = calculateStd(K_old, row_or_col)
        K_prime = scaleColGaussian(K = K_old.clone(), mean = mean, std = funScale(rc[idx], std, shift), idx = idx)

    K_new = K_prime.clone() / K_prime.sum(axis=0)

    # Compute acceptance ratio
    prob_new = 1.0
    for j in range(n_col):
        prob_new = prob_new * dirichlet.pdf(K_new[:,j], alphas)
    
    hastings_ratio = compute_ratio(K_old, K_prime.clone(), std, shift, row_or_col, idx)

    ratio = hastings_ratio * prob_new / prob_old 
    rv = random.random()
    if ratio > rv:
        acceptFlag = True
        return K_new.clone(), prob_new, acceptFlag
    else:
        return K_old.clone(), prob_old, acceptFlag

    


    
def monteCarloOneSweep(K, prob, alphas, lam, mean, std, shift, acceptCounter):
    
    for row_or_col in ['row']:
        if row_or_col == 'row':
            n = len(K)
        elif row_or_col == 'col':
            n = len(K[0])
        for i in range(n-1, -1, -1):
            K, prob, acceptFlag = monteCarloOneStep_changingStd(K, prob, alphas, lam, mean, std, shift, row_or_col, i)
            if acceptFlag:
                acceptCounter = acceptCounter + 1
            
    return K.clone(), prob, acceptCounter

    
    
def monteCarlo(T, alphas, lam, mean, std, shift, max_iter, burnInFlag, num_lag, K_old = None, prob_old = None):
    
    '''
    Initialization
    '''
    data_K = []
    acceptCounter = 0
    lowerBound = 1e-5

    if K_old is None:
        K_old = T.clone()
    n_row, n_col = len(K_old), len(K_old[0])
    if len(alphas) == 1:
        alphas = [alphas[0] for _ in range(n_row)]
    elif len(alphas) != n_row:
        print('ERROR: check alphas size, should be the same as n_row of T!')
        return
    else:
        print('ERROR: check alphas definition!')
        return

    if prob_old is None:
        prob_old = 1.0
        for j in range(n_col):
            prob_old = prob_old * dirichlet.pdf(K_old[:, j], alphas)

    # print('Prob. old: %1.2e' % prob_old)
    
    for i in tqdm(range(max_iter)):       
        K_new, prob_new, acceptCounter = monteCarloOneSweep(K_old.clone(), prob_old, alphas, lam, mean, std, shift, acceptCounter)
        if not burnInFlag:
            if i % num_lag == 0 and not burnInFlag:
                data_K.append(K_new)
        else:
            data_K.append(K_new)
        K_old, prob_old = K_new.clone(), prob_new

    # print('Prob. new: %1.2e' % prob_new)
            
    return K_old, prob_old, data_K, float(acceptCounter)/float(max_iter)/(n_row+n_col)
        

def runs(T, alphas, lam, mean, std, shift, num_burn_in, num_sampling, num_lag, K = None):
    nr, nc = len(T), len(T[0])
    data_K = []

    print("Burn in steps:")
    burnInFlag = True
    K_burn_in, prob_burn_in, data_K_burn_in, acceptRatio = monteCarlo(T, alphas, lam, mean, std, shift, num_burn_in, burnInFlag, num_lag)

    print("Burn in acceptance Ratio: %1.2f" % acceptRatio)

    print("Sampling steps:")
    burnInFlag = False

    _, _, data_K_tmp, acceptRatio = monteCarlo(T, alphas, lam, mean, std, shift, int(num_sampling), burnInFlag, num_lag, K_burn_in, prob_burn_in)
    data_K = data_K + data_K_tmp

    print("Sampling acceptance Ratio: %1.2f" % acceptRatio)
    print('Size of data K: %i' % len(data_K))

    return K, data_K, data_K_burn_in, acceptRatio



if __name__ == '__main__':
    pass

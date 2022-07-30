import sys
sys.path.append('../SRC/')
import PIOT_imports
import csv
import pandas as pd
import numpy as np
import torch
import CoDaS_PIOT_general_prior_C as PIOT
import itertools
import Sinkhorn
import matplotlib.pylab as plt
import matplotlib
import pickle
import os
import time
from scipy.stats import dirichlet

def MCMC_sim(T_1, alpha, lam, std, shift, num_burn_in = 10000, num_sampling = 1000000, num_lag = 200):

    mean = 0.0
    std = std
    
    nr, nc = len(T_1), len(T_1[0])
    
    alphas = alpha.reshape((-1,))

    K_1, data_K_1, data_K_burn_in, acceptRatio_1 = \
    runs(T_1, alphas, lam,  mean, std, shift, num_burn_in, num_sampling, num_lag)

    return data_K_1, data_K_burn_in

def sim(T, alpha, shift, lam, num_burn_in, num_lag, num_samples):
    std = 0.0
    power_k = 3
    # shift = 0.1
    # lam = 320.0
    # num_burn_in = 10000
    # num_lag = 1000
    num_sampling = num_lag*num_samples

    data_C, data_C_burn_in = MCMC_sim(T, alpha, lam, std, shift, num_burn_in, num_sampling, num_lag)
    return data_C, data_C_burn_in


def save_data(s, nr, nc, folder_path):

	filename = folder_path + 'run_time_{}_{}.csv'.format(nr, nc)
	with open(filename, 'w+') as file:
		pickle.dump(s, file, protocol=pickle.HIGHEST_PROTOCOL)

	return

def generate_C(nr, nc):
	C = torch.ones((nr, nc), dtype=torch.double)*1e-7/nr/nc
	for i in range(nr):
		for j in range(nc):
			if i != j:
				C[i][j] = np.power(np.abs(i-j)/max(nr, nc), 2)
	return C

def monteCarlo(T, alphas, lam, mean, std, shift, max_iter, burnInFlag, num_lag, C_old = None, prob_old = None):
    
    '''
    Initialization
    '''
    data_C = []
    acceptCounter = 0
    lowerBound = 1e-5

    if C_old is None:
        K_old = T
        C_old = -np.log(K_old)/lam
        C_old = C_old / C_old.sum()
    else:
        K_old = np.exp(-lam*C_old)
    n_row, n_col = len(K_old), len(K_old[0])


    if prob_old is None:
        prob_old = 1.0
        prob_old = prob_old * dirichlet.pdf( C_old.reshape((-1,))/ C_old.sum(), alphas)

    # print('Prob. old: %1.2e' % prob_old)
    
    for i in range(max_iter):       
        K_new, prob_new, acceptCounter = PIOT.monteCarloOneSweep(K_old, prob_old, alphas, lam, mean, std, shift, acceptCounter)
        if not burnInFlag:
            if i % num_lag == 0 and not burnInFlag:
                data_C.append(-np.log(K_new)/lam)
        else:
            data_C.append(-np.log(K_new)/lam)
        K_old, prob_old = K_new, prob_new

    # print('Prob. new: %1.2e' % prob_new)
            
    return -np.log(K_old)/lam, prob_old, data_C, float(acceptCounter)/float(max_iter)

def runs(T, alphas, lam, mean, std, shift, num_burn_in, num_sampling, num_lag, C = None):
    nr, nc = len(T), len(T[0])
    data_C = []

    burnInFlag = True
    C_burn_in, prob_burn_in, data_C_burn_in, acceptRatio = monteCarlo(T, alphas, lam, mean, std, shift, num_burn_in, burnInFlag, num_lag)

    return C, data_C, data_C_burn_in, acceptRatio

def main():
	with open(sys.argv[1], 'r') as my_file:
		inputs = my_file.read().split(',')
	if len(inputs) != 3:
		print('ERROR: check input arguments!')
	else:
		nr = int(inputs[0])
		nc = int(inputs[1])
	# 	value = int(inputs[2])
	# 	shift = float(inputs[3])
	# 	lam = float(inputs[4])
	# 	num_burn_in = int(inputs[5])
	# 	num_lag = int(inputs[6])
	# 	num_samples = int(inputs[7])
	# 	cnt = int(inputs[8])
		save_folder = inputs[2]

	# 	data_C_all = {}
	# 	data_C_burn_in_all = {}

	# Parameters ------------------------
	#nr, nc = 1000, 1000
	z = 1.0
	alpha = torch.tensor([[z for _ in range(nr)] for _ in range(nc)])
	shift = 0.1
	lam = 1.
	num_burn_in = 1000
	num_lag = 1
	num_samples = 1
	# -----------------------------------


	C = generate_C(nr, nc)
	T = Sinkhorn.sinkhorn_torch(np.exp(-C.clone()))

	print('Starting MCMC...')

	t1 = time.time()

	data_C, data_C_burn_in = sim(T.clone(), alpha, shift, lam, num_burn_in, num_lag, num_samples)

	t2 = time.time()

	print('Run time: {}'.format(t2-t1))

	if not os.path.exists(os.path.dirname(save_folder)):
		os.makedirs(save_folder)

	save_data(t2-t1, nr, nc, save_folder)

	print('Done...')

	return


if __name__ == "__main__":
	main()
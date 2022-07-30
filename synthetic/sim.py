import sys
sys.path.append('../SRC/')
import os
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import torch
import CoDaS_Sinkhorn
import CoDaS_PIOT
from tqdm import tqdm
import pickle
import random
import copy

def genRandomMatrix(op, size, reg = 1.0):

	print('--', op, '--')

	if op[0] == 'Uniform':
		T, r, c = CoDaS_Sinkhorn.genM(size, size)

	elif op[0] == 'Dirichlet':
		alpha = op[1]
		T, r, c = CoDaS_Sinkhorn.genM_Dirichlet(size, size, alpha)

	elif op[0] == 'Gaussian':
		mean = op[1]
		std = op[2]
		T, r, c = CoDaS_Sinkhorn.genM_Normal(size, size, mean, std)

    ''''''
#    K = torch.exp(-reg*T.clone())
#     r = r / r.sum()
#     c = c / c.sum()
        
	stopThr = 1e-9
	numItermax = 10000

	P = CoDaS_Sinkhorn.sinkhorn_torch(mat=T.clone(), row_sum = r, col_sum = c, \
                                            epsilon=stopThr, max_iter=numItermax)
    
	print("T")
	print(T, '\n')
	print("P")
	print(P, '\n')
    
    return T, P, r, c

def save_data(data_K_1, T, P, alpha, std, size):

	data = [data_K_1, T, P]

	filename = 'data_{}_{}_{}.pkl'.format(str(size), str(alpha), str(std))

	with open(filename, 'wb') as file:
		pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

	return 


def main():
	args = sys.argv
	if len(args[1:]) != 3:
		print('ERROR: check input arguments!')
	else:

		# Hyperparameters
		alpha = float(args[1]) #1.0 
		alphas = [alpha]
		mean = 0.0
		std = float(args[2]) # 0.01
		size = args[3]

		# MC parameters
		# num_burn_in * num_restart / 1000 samples
		num_burn_in = 1000
		num_sampling = 10000
		num_restart = 50

		op = ['Dirichlet', alpha]
		T, P, r, c = genRandomMatrix(op, size)

		K_1, data_K_1, acceptRatio_1 = CoDaS_PIOT.runs(P, alphas, mean, std, num_burn_in, num_sampling, num_restart)

		save_data(data_K_1, P, T, alpha, std, size)

		print('Done...')

	return


if __name__ == "__main__":
	main()
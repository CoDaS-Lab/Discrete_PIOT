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

def MCMC_sim(T_1, alpha, lam, std, shift, num_burn_in = 10000, num_sampling = 1000000, num_lag = 200):

    mean = 0.0
    std = std
    
    nr, nc = len(T_1), len(T_1[0])
    
    alphas = alpha.reshape((-1,))

    K_1, data_K_1, data_K_burn_in, acceptRatio_1 = \
    PIOT.runs(T_1, alphas, lam,  mean, std, shift, num_burn_in, num_sampling, num_lag)

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


def read_EU_T():
	folder_path = './EU_migration_data/'

	file_1 = 'EU_migration_data_1.tsv'
	df_1 = pd.read_csv(folder_path + file_1, sep = ' ')

	file_2 = 'EU_migration_data_2.tsv'
	df_2 = pd.read_csv(folder_path + file_2, sep = ' ')
	df = pd.concat([df_1, df_2.drop(labels = 'Origin', axis = 1)], axis = 1)

	countries = ['DK', 'DE', 'NL', 'CZ', 'BE', 'LU', 'FR', 'CH', 'AT']
	countries.sort()

	data = df.loc[df['Origin'].isin(countries)]
	data = data[['Origin']+countries].reindex()

	T_g = torch.tensor(data[countries].values, dtype=float)

	return T_g

def normalizeFactor(T, l):
    m, n = len(T), len(T[0])
    return np.exp( (l + np.log(T).sum()) /m/n )

def addNoise(T, noise, r, c, l):
    T_p = T.clone()
    T_p[r][c] = T_p[r][c] + noise
    a = normalizeFactor(T_p, l)
    T_p = T_p / a
    return T_p.clone(), a


def save_data(data, data_burn_in, data_plan, folder_path, cnt):

	filename = folder_path + 'data_C_{}.pkl'.format(cnt)
	with open(filename, 'wb') as file:
		pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

	filename = folder_path + 'data_C_burn_in_{}.pkl'.format(cnt)
	with open(filename, 'wb') as file:
		pickle.dump(data_burn_in, file, protocol=pickle.HIGHEST_PROTOCOL)

	filename = folder_path + 'data_plan_{}.pkl'.format(cnt)
	with open(filename, 'wb') as file:
		pickle.dump(data_plan, file, protocol=pickle.HIGHEST_PROTOCOL)

	return 

def compute_plan(data, r, c, lam):
	stopThr = 1e-9
	numItermax = 10000

	for key in data:
		cost_all = data[key]

	T_all = []

	for cost in cost_all:
		K = np.exp(-lam*cost)
		T = Sinkhorn.sinkhorn_torch(mat = K.clone(), \
			row_sum = r, \
			col_sum = c, \
			epsilon = stopThr, \
			max_iter = numItermax)
		T_all.append(T)

	return T_all


def main():
	with open(sys.argv[1], 'r') as my_file:
		inputs = my_file.read().split(',')
	if len(inputs) != 10:
		print('ERROR: check input arguments!')
	else:
		idx_i = int(inputs[0])
		idx_j = int(inputs[1])
		value = int(inputs[2])
		shift = float(inputs[3])
		lam = float(inputs[4])
		num_burn_in = int(inputs[5])
		num_lag = int(inputs[6])
		num_samples = int(inputs[7])
		cnt = int(inputs[8])
		save_folder = inputs[9]

		data_C_all = {}
		data_C_burn_in_all = {}

		z = 1.0
		alpha = torch.tensor([[z, z, z, z, z, z, z, z, z] for _ in range(9)])
		for i in range(len(alpha)):
		    alpha[i][i] = 25.

		noises = [0.0]

		T_g = read_EU_T()

		marg = T_g.clone()

		for i in range(len(T_g)):
			T_g[i][i] = 1.
		T_g[idx_i][idx_j] = value

		for noise in noises:
		    T_p, _ = addNoise(T_g, noise, idx_i, idx_j, lam)
		    print(T_p)
		    data_C, data_C_burn_in = sim(T_p.clone(), alpha, shift, lam, num_burn_in, num_lag, num_samples)
		    data_C_all[str(noise)] = data_C
		    data_C_burn_in_all[str(noise)] = data_C_burn_in_all

		predicted_plan = compute_plan(data_C_all, marg.sum(axis=1), marg.sum(axis=0), lam)

		if not os.path.exists(os.path.dirname(save_folder)):
			os.makedirs(save_folder)

		save_data(data_C_all, data_C_burn_in_all, predicted_plan, save_folder, cnt)

		print('Done...')

	return


if __name__ == "__main__":
	main()
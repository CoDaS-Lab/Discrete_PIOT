import sys
sys.path.append('../SRC')
import PIOT
import plot
import Sinkhorn
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pickle
import itertools
import matplotlib
from scipy.stats import dirichlet
matplotlib.rcParams.update({'font.size': 20})

def MCMC_sim(T_1, alpha, std, power_k, shift, W, num_burn_in = 10000, num_sampling = 1000000, num_lag = 200):

    mean = 0.0
    std = std
    
    nr, nc = len(T_1), len(T_1[0])
    
    alphas = [alpha]
    
    print(T_1)
    K_1, data_K_1, data_K_burn_in, acceptRatio_1 = \
        PIOT.runs(T_1, alphas, mean, std, power_k, shift, W, num_burn_in, num_sampling, num_lag)

    print('Done...')
    return data_K_1, data_K_burn_in

def autocorr(x,lags,var):
    n_vectors = len(x)
    nr, nc = len(x[0]), len(x[0][0])
    mean = x.mean(axis = 0)
    xp = torch.stack([row-mean for row in x])
    corr = np.array([np.correlate(xp[:,r,c],xp[:,r,c],'full') \
                     for r, c in itertools.product(range(nr), range(nc))])[:, n_vectors-1:]
    div = np.array([n_vectors-i for i in range(len(lags))])
    acorr = corr.sum(axis=0)[:len(lags)]/var/div

    return acorr[:len(lags)]

def plot_row_sum_corr(data):
    nc = len(data[0][0])
    row_sum = []

    for i in range(len(data)):
        row_sum.append(np.array(data[i].sum(axis=1)))

    lags = range(1000)
    var = np.var(row_sum, axis = 0).sum()
    corr = plot.autocorr(row_sum, lags, var)

    x = range(len(lags))
    plus_1_div_exp = 1/np.exp(1)*np.ones(len(lags))
    minus_1_div_exp = - plus_1_div_exp

    plt.plot(lags, corr)
    plt.plot(x, plus_1_div_exp, 'k--')
    plt.plot(x, minus_1_div_exp, 'k--')
    
def plot_corr(data, lag = 1000, save = False, path = './'):
    
    lags = range(lag)
    var = torch.sum(torch.var(data, axis = 0))
    corr = autocorr(data, lags, var)
    
    x = range(len(lags))
    plus_1_div_exp = 1/np.exp(1)*np.ones(len(lags))
    minus_1_div_exp = - plus_1_div_exp

    fig = plt.figure()

    plt.plot(lags, corr)
    plt.plot(x, plus_1_div_exp, 'k--')
    plt.plot(x, minus_1_div_exp, 'k--')
    plt.xlabel('t')
    plt.ylabel("R(t)")
    if save == True:
        plt.savefig(path)
        
def plot_one_matrix_simplex(M):
    plot.plot_points(np.array([t[0] for t in M]), 'r', 5)
    plot.plot_points(np.array([t[1] for t in M]), 'g', 5)
    plot.plot_points(np.array([t[2] for t in M]), 'b', 5)
    
def plot_samples(data, bw_method = 0.1):
    nr = len(data[0])
    nc = len(data[0][0])
    x_i = [[] for _ in range(nr)]
    df_all = pd.DataFrame()

    colors = ['r', 'g', 'b']

    for i in range(nr):
        for j in range(nc):
            x_i[i] = np.array([K[i][j].numpy() for K in data])


            df = pd.DataFrame(x_i[i], columns = ['({},{})'.format(i, j)])

            ax1 = df.plot.density(bw_method=bw_method)
            ax1.set_xlim(0, 1)
            ax1.set_ylim(bottom=0)

    
def plot_samples_column(data, bw_method = 0.1):
    nr = len(data[0])
    nc = len(data[0][0])
    x_i = [[] for _ in range(nr)]
    df_all = pd.DataFrame()

    colors = ['r', 'g', 'b']

    for j in range(nc):
        x_i = np.array([[K[i][j].numpy() for i in range(nr)] for K in data])

        df = pd.DataFrame(x_i, columns = ['({},{})'.format(i,j) for i in range(nr) ])

        ax1 = df.plot.density(bw_method=bw_method)
        ax1.set_xlim(0, 1)

        
def running_average(data):
    
    nc = len(data[0][0])
    x = []

    for i in range(len(data)):
        x.append(np.array(data[i].sum(axis=1)))

    x = np.array(x)
    nr = len(x)
    nc = len(x[0])
    ra = [x[0]]
    for i in range(1, nr):
        ra.append(ra[i-1] + x[i])
    for i in range(1, nr):
        ra[i] /= (i+1)
    return ra

def plot_row_sum_running_average(data, save = False, path = './'):
    ra = running_average(data)
    df_ra = pd.DataFrame(ra)
    ax = df_ra.plot()
    ax.set_ylabel('Row sum')
    ax.set_xlabel('# samples')
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    #ax.set_ylim(bottom=0.2)
    if save == True:
        ax.figure.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')


def normalizeFactor(T, lam):
    m, n = len(T), len(T[0])
    return np.exp( (lam + np.log(T).sum()) /m/n )


            
def plot_pdf(data, bw_method = 0.05, save = False, path = '/'):


    nr = len(data[0])
    nc = len(data[0][0])

    colors = ['r', 'g', 'b']

    idx = 0

    for i in range(nr):
        for j in range(nc):
            x_i = np.array([K[i][j].numpy() for K in data])
            df = pd.DataFrame(x_i, columns = ['({},{})'.format(i+1,j+1)])
            ax1 = df.plot.density(bw_method=bw_method, color='k')
            if save == True:
                save_path = path +'_cost_'+str(idx)+'.pdf'
                ax1.figure.savefig(save_path)
                idx += 1

def save_MCMC_figs(data_K_burn_in, data_K, save_folder, file_name):

    save_path = save_folder+file_name+'_corr.pdf'
    plot_corr(torch.stack(data_K_burn_in), lag = 1000, save = True, path = save_path)
    

    save_path = save_folder+file_name+'_RA.pdf'
    plot_row_sum_running_average(data_K, save = True, path = save_path)

    save_path = save_folder+file_name
    plot_pdf(data_K, save = True, path = save_path)

    return

    
def data_to_mat(data, cols = ['PTS','ORB','DRB','AST','STL']):
    T =torch.tensor( data[cols].to_numpy() )
    return T
    

if __name__ == '__main__':
    pass
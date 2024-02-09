# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:19:14 2024

@author: paul-
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
import matplotlib.pyplot as plt
import yfinance as yf
np.random.seed(0) #avant 0
from numba import jit, prange
import matplotlib.gridspec as gridspec


@jit(nopython=True)
def kernel_function(u):
    return np.where(np.abs(u) < 1, (3/4) * (1 - u**2), 0)

@jit(nopython=True)
def init_pdf(x, X, w, h, t0):
    u = (x-X[:t0])/h
    delta = np.array([t0-i for i in range(t0)])
    f = np.dot(w**delta, kernel_function(u))*(1-w)/((1-w**t0)*h)
    return f

@jit(nopython=True)
def dynamic_pdf(x, X, h, w, t0):
    f = []
    f0 = init_pdf(x, X, w, h, t0)
    f.append(f0)
    u = (x-X[t0+1:])/h
    n = len(u)
    for i in range(n):
        ft = w*f[-1] + (1-w)/h * kernel_function(u[i])
        f.append(ft)
    return f

@jit(nopython=True)
def kernel_prim(u):
    return np.where(u <= -1, 0, np.where(u >= 1, 1, (3/4) * (u - (1/3)*u**3) + 0.5))

@jit(nopython=True)
def init_kernel_CDF(x, X, h, w, t0):
    delta = np.array([t0-i for i in range(t0+1)])
    u = (x-X[:t0+1])/h
    
    F = np.dot(w**delta, kernel_prim(u))*(1-w)/(1-w**t0)
    return F 

def dynamic_kernel_CDF(x, X, h, w, t0):
    F = []
    F0 = init_kernel_CDF(x, X, h, w, t0)
    F.append(F0)
    u = (x-X[t0+1:])/h
    n = len(u)
    for i in range(n):
        Ft = w * F[-1] + (1-w) * kernel_prim(u[i])
        F.append(Ft)
    return np.array(F)

def PIT(x, X, t0, F, C = False, p=0.05):
    Z = []
    for i in range(t0+1,len(X)):
        idx = int(np.where(x==X[i])[0])
        z =  F[:,i-t0-1][idx]
        if C:
            if (z <= p) or (z >= 1-p):
                Z.append(z)
        else:
            Z.append(z)
    return Z

def KS_test(PIT):
    Z = np.array(PIT)
    n = len(Z)
    
    stat_opt = -np.inf
    for s in range(n):
        nb = 0
        for u in range(n):
            nb += (Z[u]<=Z[s])
        stat = abs(Z[s] - nb/(n+1)) #+1
        if stat > stat_opt:
            stat_opt = stat
            
    return stat_opt

def KS_test_tau(PIT, tau):
    Z = np.array(PIT)
    n = len(Z)
    
    stat_opt = -np.inf
    for s in range(n-tau):
        nb = 0
        for u in range(n-tau):
            nb += (Z[u]<=Z[s] and Z[u] >= 0)*(Z[u+tau]<=Z[s+tau] and Z[u+tau]>=0)
        stat = abs(Z[s]*Z[s+tau] - nb/(n-tau+1)) #-tau+1
        if stat > stat_opt:
            stat_opt = stat
            
    return stat_opt


def dv(PIT, t0, v):
    opt_stat = KS_test(PIT) * np.sqrt(len(PIT))
    for tau in range(1,v+1):
        stat = KS_test_tau(PIT, tau) * np.sqrt(len(PIT)-tau)
        if stat >opt_stat:
         opt_stat = stat
    return opt_stat

        
def optimiser(rets,t0,v,h,w,C):
    
    if w < 1:
        x_range = np.sort(rets)
        F = []
        for x in x_range:
            Fx = dynamic_kernel_CDF(x, rets, h, w, t0)
            F.append(Fx)
        F = np.array(F)
        Z = PIT(x_range, rets, t0, F, C)
        
        return dv(Z,t0,v)
    else:
        return np.inf

def find_optimal_parameters(rets, t0, v=22, constraint = True, C = False):

    def objective_function(params):
        h, w = params
        return optimiser(rets, t0, v, h, w, C)

    initial_params = [0.0045, 0.98]
    bounds = [(1e-8, None),(1-v**(-1), 1)] if constraint else  [(1e-8, None),(0, 1)] 
    options = {'eps': 1e-6 }
    result = minimize(objective_function, initial_params, method='L-BFGS-B', jac='3-point', bounds=bounds, options=options)
    optimal_h, optimal_w = result.x
    print(result)
    print(result.success)
    print(result.message)
    
    return optimal_h, optimal_w

def KS_divergence(F):
    F0 = F[:,0]
    div =[]
    
    for t in range(1,F.shape[1]):
        div.append(np.max(abs(F[:,t] - F0)))
        
    return div

def Hellinger_divergence(f):
    f0 = f[:,0]
    div=[]
    
    for t in range(1,f.shape[1]):
        div_t_t0 = np.sqrt(0.5*np.sum((np.sqrt(f[:,t]) - np.sqrt(f0))**2)/(f.shape[0]))
        div.append(div_t_t0/2)
        
    return div

def W_divergence(F):
    F0 = F[:,0]
    div = []
    
    for t in range(1,F.shape[1]):
        div.append(np.sum(abs(F[:,t] - F0)/(F.shape[0])))
    
    return div
    
def KL_divergence(f):
    f0 = f[:,0]
    div=[]
    
    for t in range(1, f.shape[1]):

        f_t = np.maximum(f[:,t], 0.1)
        f0_safe = np.maximum(f0, 0.1)

        div_t_t0 = np.sum(f_t * np.log(f_t / f0_safe))/(f.shape[0])
        div.append(div_t_t0)
        
    return np.array(div)/7

def generate_time_varying_cauchy_distribution(n=2000):
    mu_dynamic = np.arange(n) / 100
    return np.random.standard_cauchy(n) + mu_dynamic

def log_likelihood(x, X, t0, F):
    f_array = []
    for i in range(t0+1,len(X)):
        idx = int(np.where(x==X[i])[0])
        f =  F[:,i-t0-1][idx]
        f = max(f, 1e-10)
        f_array.append(f)

    return np.sum(np.log(np.array(f_array)))

def optimiser_MLE(rets,t0,h,w):    
    if w < 1:
        x_range = np.sort(rets)
        F = []
        for x in x_range:
            Fx = dynamic_pdf(x, rets, h, w, t0)
            F.append(Fx)
        F = np.array(F)
        return log_likelihood(x_range, rets, t0, F)


def find_optimal_parameters_MLE(rets, t0=1000):
 
    def objective_function(params):
         h, w = params
         Z = optimiser_MLE(rets, t0, h, w)
         
         return -Z if Z else np.inf

    initial_params = [0.5, 0.9] #0.85
    bounds = [(1e-8, None),(1e-8, 1)] 
#    options = {'eps': 1e-6 }
    result = minimize(objective_function, initial_params, method='L-BFGS-B', bounds=bounds)
    optimal_h, optimal_w = result.x
    print(result)
    print(result.success)
    print(result.message)
    
    return optimal_h, optimal_w

def simulate_brownian_motion(num_points, num_paths=10, drift=0, volatility=1):
    dt = 1 / num_points
    paths = np.cumsum(np.random.normal(drift * dt, volatility * np.sqrt(dt), (num_points, num_paths)), axis=0)
    return paths

def dynamic_kernel_CDF_b(x, X, h, w, F0):
    F = []
    F.append(F0)
    u = (x-X[1:])/h
    n = len(u)
    for i in range(n):
        Ft = w * F[-1] + (1-w) * kernel_prim(u[i])
        F.append(Ft)
    return np.array(F)

def dynamic_pdf_b(x, X, h, w, f0):
    f = []
    f.append(f0)
    u = (x-X[1:])/h
    n = len(u)
    for i in range(n):
        ft = w*f[-1] + (1-w)/h * kernel_function(u[i])
        f.append(ft)
    return f

def calculate_statistics_for_paths(paths, t0, h, w):
    
    F_dict = {}
    f_dict = {}

    for i in range(paths.shape[1]): 
        F_temp = []
        f_temp = []

        for x in np.sort(paths[:, i]):
            F_temp.append(dynamic_kernel_CDF(x, paths[:, i], h, w, t0))
            f_temp.append(dynamic_pdf(x, paths[:, i], h, w, t0))

        F_dict[i] = np.array(F_temp)
        f_dict[i] = np.array(f_temp)
    
        
    all_stats = {'KS': {}, 'Hellinger': {}, 'Wasserstein': {}, 'KL': {}}

    for i in F_dict:  

        KS = KS_divergence(F_dict[i])
        Hellinger = Hellinger_divergence(f_dict[i])
        Wasserstein = W_divergence(F_dict[i])
        KL = KL_divergence(f_dict[i])

        all_stats['KS'][i] = KS
        all_stats['Hellinger'][i] = Hellinger
        all_stats['Wasserstein'][i] = Wasserstein
        all_stats['KL'][i] = KL

    return all_stats

def calculate_quantiles(all_stats, quantiles=[0.95, 0.99, 0.999]):
    quantile_values = {key: {q: [] for q in quantiles} for key in all_stats}

    for stat in all_stats:
        num_points = len(next(iter(all_stats[stat].values())))

        for q in quantiles:
            for t in range(num_points):
                data_at_t = [all_stats[stat][i][t] for i in all_stats[stat]]
                quantile_values[stat][q].append(np.quantile(data_at_t, q))

    return quantile_values
  
def calculate_quantiles(all_stats, quantiles=[0.95, 0.99, 0.999]):
    quantile_values = {key: {q: [] for q in quantiles} for key in all_stats}

    quantile_d = {0.95: 5, 0.99: 4, 0.999: 3}

    for stat in all_stats:
        num_points = len(next(iter(all_stats[stat].values())))

        for q in quantiles:
            for t in range(num_points):
                data_at_t = [all_stats[stat][i][t] for i in all_stats[stat]]
                quantile_value = np.quantile(data_at_t, q)
                adjusted_quantile_value = quantile_value / quantile_d[q]
                quantile_values[stat][q].append(adjusted_quantile_value)

    return quantile_values

def calculate_densities(x_range, X, h, w, t0_index):
    f = []
    for x in x_range:
        fx = dynamic_pdf(x, X, h, w, t0_index)
        f.append(fx)
    return np.array(f)

def calculate_CDF(x_range, X, h, w, t0_index):
    F = []
    for x in x_range:
        Fx = dynamic_kernel_CDF(x, X, h, w, t0_index)
        F.append(Fx)
    return np.array(F)

def dynamic_cauchy_densitie(x):
    mu = np.array([t / 100 for t in range(1000, 2000)])
    return 1 / (np.pi * (1 + (x - mu)**2))

def dynamic_cauchy_CDF(x):
    mu = np.array([t / 100 for t in range(1000, 2000)])
    return 1/np.pi * np.arctan(x - mu) + 0.5

def brownian_density(x, t0, sigma=1):
    t_values = np.array([t for t in range(141)])
    return 1 / np.sqrt(2 * np.pi * sigma**2 * t_values) * np.exp(-x**2 / (2 * sigma**2 * t_values))

def brownian_CDF(x, t0, sigma=1):
    t_values = np.array([t for t in range(141)])
    return 0.5 * (1 + erf(x / (np.sqrt(2 * sigma**2 * t_values))))

def KS_divergence_sim(F, F_true):
    div =[]
    
    for t in range(1,F.shape[1]-1):
        div.append(max(abs(F[:,t] - F_true[:,t+1])))
        
    return div

def Hellinger_divergence_sim(f, f_true):
    div=[]
    
    for t in range(1,f.shape[1]):
        div_t_t0 = np.sqrt(0.5*np.sum((np.sqrt(f[:,t]) - np.sqrt(f_true[:,t]))**2)/(f.shape[0]))
        div.append(div_t_t0/2)
        
    return div

def W_divergence_sim(F, F_true):
    div = []
    
    for t in range(1,F.shape[1]):
        div.append(np.sum(abs(F[:,t] - F_true[:,t])/(F.shape[0])))
    
    return div
    
def KL_divergence_sim(f, f_true):
    div=[]
    
    for t in range(1,f.shape[1]):
        
        fhat_t = np.maximum(f[:,t], 0.1)
        f_t = np.maximum(f_true[:,t], 0.1)
        
        div_t_t0 = np.sum(fhat_t * np.log(fhat_t/f_t))
        div.append(abs(div_t_t0))
        
    return np.array(div)/(7*20)*0.8

################## chargementdes données ##################




t0_date = '2019-11-01'
t1_date = '2020-02-07'
indices = ['^GSPC', '^STOXX50E', '^KS11']
data = yf.download(indices, start='2015-01-01', end='2020-05-31')['Adj Close']
data_filled = data.fillna(method='ffill')
data_filled = data_filled[1:]


t0_date = '2019-11-01'
t1_date = '2022-03-04'
t2_date = '2023-10-20'
indices = ['^GSPC', '^STOXX50E', '^KS11']
data = yf.download(indices, start='2015-01-01', end='2024-01-01')['Adj Close']
data_filled = data.fillna(method='ffill')
data_filled = data_filled[1:]

t0_index = data.index.get_loc(t0_date)
t0_index = t0_index + 1
t1_index = data.index.get_loc(t1_date) -2 -t0_index
t2_index = data.index.get_loc(t2_date)-2 -t0_index

prices_SP = data_filled['^GSPC'].values
prices_ST = data_filled['^STOXX50E'].values
prices_KS = data_filled['^KS11'].values

rets_SP = np.log(prices_SP[1:]) - np.log(prices_SP[:-1])
rets_SP = rets_SP[1:]

rets_KS = np.log(prices_KS[1:]) - np.log(prices_KS[:-1])
rets_KS = rets_KS[1:]

rets_ST = np.log(prices_ST[1:]) - np.log(prices_ST[:-1])
rets_ST = np.array(rets_ST[1:])

################## OPTIMISATION ##################
##################################################

h_SNP_C, w_SNP_C = find_optimal_parameters(rets_SP, t0_index)   # []

h_SNP_C, w_SNP_C = find_optimal_parameters(rets_ST, t0_index)


h_KS_C, w_KS_C = find_optimal_parameters(rets_KS, t0_index) # [0.04011837, 0.77371585]
h_KS_C, w_KS_C = [0.04011837, 0.77371585]

h_ES_C, w_ES_C = find_optimal_parameters(rets_SP, t0_index) # [0.07354711, 0.7838567]
h_ES_C, w_ES_C = [0.07354711, 0.7838567]

h_SNP_NC, w_SNP_NC = find_optimal_parameters(rets_SP, t0_index, constraint=False)  # [1e-5, 0.94572377]
h_SNP_NC, w_SNP_NC = [1e-5, 0.94572377]

h_KS_NC, w_KS_NC = find_optimal_parameters(rets_KS, t0_index, constraint=False) # []
h_ES_NC, w_ES_NC = find_optimal_parameters(rets_SP, t0_index, constraint=False) # [1e-5, 0.827030538]
h_ES_NC, w_ES_NC = [1e-5, 0.827030538]

################## OPTIMISATION sur simulations ##################
#################################################################
Y = generate_time_varying_cauchy_distribution()


h_mle_sim,w_mle_sim = find_optimal_parameters_MLE(Y)
h_star_sim,w_star_sim = find_optimal_parameters(Y, 1000, constraint=False)
h_C_sim,w_C_sim = find_optimal_parameters(Y, 1000, constraint=False, C=True)

#h_mle,w_mle = find_optimal_parameters_MLE(rets, t0_index)
# [4.173e-02, 7.214e-01]
# 4.4e-2
#0.838

####

test = optimiser_MLE(rets, t0_index, h_mle, w_mle)

optimiser_MLE(rets, t0_index, 4.4e-2, 0.838)

################## recuperation des densitées + plots ##################
########################################################################

x_range_SP = np.linspace(np.min(rets_SP), np.max(rets_SP), 1000)
x_range_ST = np.linspace(np.min(rets_ST), np.max(rets_ST), 1000)
x_range_KS = np.linspace(np.min(rets_KS), np.max(rets_KS), 1000)
x_range = np.linspace(-0.12,0.12,1000)

w = 0.955
h = 0.012

f_array_SP = calculate_densities(x_range_SP, rets_SP, h, w, t0_index)
F_array_SP = calculate_CDF(x_range_SP, rets_SP, h, w, t0_index)

f_array_ST = calculate_densities(x_range_ST, rets_ST, h, w, t0_index)
F_array_ST = calculate_CDF(x_range_ST, rets_ST, h, w, t0_index)

f_array_KS = calculate_densities(x_range_KS, rets_KS, h, w, t0_index)
F_array_KS = calculate_CDF(x_range_KS, rets_KS, h, w, t0_index)



###############calcul des stats + plots#################
########################################################

ks_SP = KS_divergence(F_array_SP)
ks_ST = KS_divergence(F_array_ST)
ks_KS = KS_divergence(F_array_KS)

W_SP = W_divergence(F_array_SP)
W_ST = W_divergence(F_array_ST)
W_KS = W_divergence(F_array_KS)

H_SP = Hellinger_divergence(f_array_SP)
H_ST = Hellinger_divergence(f_array_ST)
H_KS = Hellinger_divergence(f_array_KS)

KL_SP = KL_divergence(f_array_SP)
KL_ST = KL_divergence(f_array_ST)
KL_KS = KL_divergence(f_array_KS)


fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(ks_SP, label='Sp')
axs[0, 0].plot(ks_ST, label='ST')
axs[0, 0].plot(ks_KS, label='KS')
axs[0, 0].set_title('KS Divergence')
axs[0, 0].legend()

axs[0, 1].plot(H_SP, label='SP')
axs[0, 1].plot(H_ST, label='ST')
axs[0, 1].plot(H_KS, label='KS')
axs[0, 1].set_title('Hellinger Divergence')
axs[0, 1].legend()

axs[1, 0].plot(W_SP, label='Sp')
axs[1, 0].plot(W_ST, label='ST')
axs[1, 0].plot(W_KS, label='KS')
axs[1, 0].set_title('Wasserstein Divergence')
axs[1, 0].legend()

axs[1, 1].plot(KL_SP, label='SP')
axs[1, 1].plot(KL_ST, label='ST')
axs[1, 1].plot(KL_KS, label='KS')
axs[1, 1].set_title('KL Divergence')
axs[1, 1].legend()

plt.tight_layout()
plt.show()


##########################
SP_max_i = np.argmax(H_SP)
ST_max_i = np.argmax(H_ST)
KS_max_i = np.argmax(H_KS)


fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :]) 

ax1.plot(x_range,f_array_ST[:,0])
ax1.plot(x_range,f_array_ST[:,t1_index])
ax1.plot(x_range,f_array_ST[:,t2_index])
ax1.plot(x_range,f_array_ST[:,ST_max_i])
ax1.plot(x_range,f_array_ST[:,140])
ax1.plot(x_range,f_array_ST[:,-1])
ax1.set_title('Eursostoxx')
ax1.legend()

ax2.plot(x_range,f_array_KS[:,0])
ax2.plot(x_range,f_array_KS[:,t1_index])
ax2.plot(x_range,f_array_KS[:,t2_index])
ax2.plot(x_range,f_array_KS[:,KS_max_i])
ax2.plot(x_range,f_array_KS[:,140])
ax2.plot(x_range,f_array_KS[:,-1])
ax2.set_title('KOSPI')
ax2.legend()

ax3.plot(x_range,f_array_SP[:,0], label = "estimated density at t0")
ax3.plot(x_range,f_array_SP[:,t1_index], label= "estimated density at 2022-03-04")
ax3.plot(x_range,f_array_SP[:,t2_index], label= "estimated density at 2023-10-20")
ax3.plot(x_range,f_array_SP[:,SP_max_i], label= "estimated density at peak")
ax3.plot(x_range,f_array_SP[:,140], label= "estimated density at 2020-05-31")
ax3.plot(x_range,f_array_SP[:,-1], label= "estimated density at last date")
ax3.set_title('SnP')
ax3.legend()

plt.tight_layout()
plt.show()



plt.plot(x_range,f_array_ST[:,0])
plt.plot(x_range,f_array_ST[:,t1_index])
plt.plot(x_range,f_array_ST[:,ST_max_i])
plt.plot(x_range,f_array_ST[:,-1])


plt.plot(x_range,f_array_KS[:,0])
plt.plot(x_range,f_array_KS[:,t1_index])
plt.plot(x_range,f_array_KS[:,KS_max_i])
plt.plot(x_range,f_array_KS[:,-1])



############ quantiles des stats à partir des browniens #####################
#############################################################################
h,w = (0.012, 0.955)
brownian_paths = simulate_brownian_motion(len(rets_SP), 500)

b_range = np.linspace(np.min(brownian_paths)-20, np.max(brownian_paths)+20, brownian_paths.shape[0])

f_B = []
for y in b_range:
    fY = brownian_density(y, t0_index)
    f_B.append(fY)
f_B = np.array(f_B)
f_B = f_B[:,1:]

F_B = []
for y in b_range:
    FY = brownian_CDF(y, t0_index)
    F_B.append(FY)
F_B = np.array(F_B)
F_B = F_B[:,1:]

all_stats = calculate_statistics_for_paths(brownian_paths,t0_index, h, w)
quantile_values = calculate_quantiles(all_stats)

quantiles_ks_95 = np.array(quantile_values['KS'][0.95])
quantiles_ks_99 = np.array(quantile_values['KS'][0.99])
quantiles_ks_999 = np.array(quantile_values['KS'][0.999])

quantiles_Hellinger_95 = np.array(quantile_values['Hellinger'][0.95])
quantiles_Hellinger_99 = np.array(quantile_values['Hellinger'][0.99])
quantiles_Hellinger_999 = np.array(quantile_values['Hellinger'][0.999])

quantiles_Wasserstein_95 = np.array(quantile_values['Wasserstein'][0.95])
quantiles_Wasserstein_99 = np.array(quantile_values['Wasserstein'][0.99])
quantiles_Wasserstein_999 = np.array(quantile_values['Wasserstein'][0.999])

quantiles_KL_95 = np.array(quantile_values['KL'][0.95])
quantiles_KL_99 = np.array(quantile_values['KL'][0.99])
quantiles_KL_999 = np.array(quantile_values['KL'][0.999])





##################################################
##################################################

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(ks_SP, label='Sp')
axs[0, 0].plot(ks_ST, label='ST')
axs[0, 0].plot(ks_KS, label='KS')
axs[0, 0].plot(quantiles_ks_95, label='95th Quantile', linestyle='--')
axs[0, 0].plot(quantiles_ks_99, label='99th Quantile', linestyle='--')
axs[0, 0].plot(quantiles_ks_999, label='99.9th Quantile', linestyle='--')
axs[0, 0].set_title('KS Divergence')
axs[0, 0].legend()

axs[0, 1].plot(H_SP, label='SP')
axs[0, 1].plot(H_ST, label='ST')
axs[0, 1].plot(H_KS, label='KS')
axs[0, 1].plot(quantiles_Hellinger_95, label='95th Quantile', linestyle='--')
axs[0, 1].plot(quantiles_Hellinger_99, label='99th Quantile', linestyle='--')
axs[0, 1].plot(quantiles_Wasserstein_999, label='99.9th Quantile', linestyle='--')
axs[0, 1].set_title('Hellinger Divergence')
axs[0, 1].legend()

axs[1, 0].plot(W_SP, label='Sp')
axs[1, 0].plot(W_ST, label='ST')
axs[1, 0].plot(W_KS, label='KS')
axs[1, 0].plot(quantiles_Wasserstein_95/5, label='95th Quantile', linestyle='--')
axs[1, 0].plot(quantiles_Wasserstein_99/5, label='99th Quantile', linestyle='--')
axs[1, 0].plot(quantiles_Wasserstein_999/5, label='99.9th Quantile', linestyle='--')
axs[1, 0].set_title('Wasserstein Divergence')
axs[1, 0].legend()

axs[1, 1].plot(KL_SP, label='SP')
axs[1, 1].plot(KL_ST, label='ST')
axs[1, 1].plot(KL_KS, label='KS')
axs[1, 1].plot(quantiles_KL_95, label='95th Quantile', linestyle='--')
axs[1, 1].plot(quantiles_KL_99, label='99th Quantile', linestyle='--')
axs[1, 1].plot(quantiles_KL_999, label='99.9th Quantile', linestyle='--')
axs[1, 1].set_title('KL Divergence')
axs[1, 1].legend()

plt.tight_layout()
plt.show()


############ plots données simulées #####################
#########################################################
Y = generate_time_varying_cauchy_distribution()
y_range = np.linspace(5, 35, 1000)


(h_star, w_star) = (0.488, 0.902)
(h_HO, w_HO) = (0.596, 0.989)
(h_C, w_C) = (1.375, 0.979)


f_star = calculate_densities(y_range, Y, h_star, w_star, 1000)
F_star = calculate_CDF(y_range, Y, h_star, w_star, 1000)

f_HO = calculate_densities(y_range, Y, h_HO, w_HO, 1000)
F_HO = calculate_CDF(y_range, Y, h_HO, w_HO, 1000)

f_C = calculate_densities(y_range, Y, h_C, w_C, 1000)
F_C = calculate_CDF(y_range, Y, h_C_sim, w_C_sim, 1000)



f_true = []
for y in y_range:
    fY = dynamic_cauchy_densitie(y)
    f_true.append(fY)
f_true = np.array(f_true)

F_true = []
for y in y_range:
    FY = dynamic_cauchy_CDF(y)
    F_true.append(FY)
F_true = np.array(F_true)

plt.plot(y_range, f_star[:,999])
plt.plot(y_range, f_HO[:,999])
plt.plot(y_range, f_C[:,999])
plt.plot(y_range, f_true[:,999])


KS_stats_sim_star = KS_divergence_sim(F_star, F_true)
KS_stats_sim_HO = KS_divergence_sim(F_HO, F_true)
KS_stats_sim_C = KS_divergence_sim(F_C, F_true)

H_stats_sim_star = Hellinger_divergence_sim(F_star, F_true)
H_stats_sim_HO = Hellinger_divergence_sim(F_HO, F_true)
H_stats_sim_C = Hellinger_divergence_sim(F_C, F_true)

W_stats_sim_star = W_divergence_sim(F_star, F_true)
W_stats_sim_HO = W_divergence_sim(F_HO, F_true)
W_stats_sim_C = W_divergence_sim(F_C, F_true)

KL_stats_sim_star = KL_divergence_sim(F_star, F_true)
KL_stats_sim_HO = KL_divergence_sim(F_HO, F_true)
KL_stats_sim_C = KL_divergence_sim(F_C, F_true)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))


axs[0, 0].plot(KS_stats_sim_star, label='Star')
axs[0, 0].plot(KS_stats_sim_HO, label='HO')
axs[0, 0].plot(KS_stats_sim_C, label='C')
axs[0, 0].set_title('KS Divergence')
axs[0, 0].legend()

axs[0, 1].plot(H_stats_sim_star, label='Star')
axs[0, 1].plot(H_stats_sim_HO, label='HO')
axs[0, 1].plot(H_stats_sim_C, label='C')
axs[0, 1].set_title('Hellinger Divergence')
axs[0, 1].legend()

axs[1, 0].plot(W_stats_sim_star, label='Star')
axs[1, 0].plot(W_stats_sim_HO, label='HO')
axs[1, 0].plot(W_stats_sim_C, label='C')
axs[1, 0].set_title('Wasserstein Divergence')
axs[1, 0].legend()

axs[1, 1].plot(KL_stats_sim_star, label='Star')
axs[1, 1].plot(KL_stats_sim_HO, label='HO')
axs[1, 1].plot(KL_stats_sim_C, label='C')
axs[1, 1].set_title('KL Divergence')
axs[1, 1].legend()

plt.tight_layout()
plt.show()


############ terrain de jeu #####################
#################################################

#test dv

h_values = [0.01, 0.001, 1.0*10**(-5), 4.4*10**(-3)]
w_values = [0.7, 0.75, 0.8, 0.864, 0.856]

dv_list = []
x_range = np.sort(rets)

for h in h_values:
    for w in w_values:
        F = []
        for x in x_range:
            Fx = dynamic_kernel_CDF(x, rets, h, w, t0_index)
            F.append(Fx)
        F_array = np.array(F)

        Z = PIT(x_range, rets, t0_index, F_array)
        dv_value = dv(Z, t0_index, 22)
        dv_list.append((h, w, dv_value))
        
print(dv_list)

max_ = np.inf
for i, stats in enumerate(dv_list):
    if stats[2] < max_:
        max_ = stats[2]
        i_max = i

dv_list[i_max]

h,w = [ 1.389e-02 , 9.859e-01] #from optimisation fro Snp500, constraint but not censored
# dv : 0.7459016045732589

h,w = [ 6.9*10**(-3) , 0.955] #from paper
# dv : 1.033796445942639

x_range = np.sort(rets)
F = []
for x in x_range:
    Fx = dynamic_kernel_CDF(x, rets, h, w, t0_index)
    F.append(Fx)
F_array = np.array(F)
Z = PIT(x_range, rets, t0_index, F_array)
dv_value = dv(Z, t0_index, 22)
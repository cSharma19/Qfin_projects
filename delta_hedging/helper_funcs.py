# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:33:10 2023

@author: local_admin
# Used in project folder Delta Hedging
"""
import numpy as np
from scipy import stats
from math import log, sqrt, exp
import pandas as pd


def simpath(S0, r, sigma, ttm, numSteps, numPaths):
    rng = np.random.default_rng(seed=123)
    
    dt = ttm/numSteps
    S = np.zeros((numSteps, numPaths))    
    S[0] = S0
    z = rng.normal(size=(numSteps, numPaths))
    
    for i in range(0, numSteps-1):
        S[i+1] = S[i] * np.exp((r - 0.5 * np.power(sigma,2)) * dt + sigma * np.power(dt, 0.5) * z[i+1, :])
        
    return S

#%%
def bs_call(S0, strike, sigma, ttm, r):
    d1 = (np.log(S0/strike) + (r + 0.5 * np.power(sigma, 2)) * ttm )/(sigma * sqrt(ttm))
    d2 = d1 - sigma * np.power(ttm, 0.5)                                                                
    bs_call = S0 * stats.norm.cdf(d1, 0, 1) - strike * exp(-r * ttm) * stats.norm.cdf(d2, 0, 1)
    # bscall_delta = stats.norm.cdf(d1, 0, 1)
    return bs_call
#%%
def bs_delta(path, strike, sigma, ttm_vec, r, numPaths):
    # d1 = (np.log(path/strike) + (r + 0.5 * np.power(sigma, 2)) * ttm_vec )/(sigma * sqrt(ttm_vec))
    d1 = (np.log(path/strike) + (r + 0.5 * np.power(sigma, 2)) * ttm_vec )/(sigma * np.power(ttm_vec, 0.5))
    # d1 = (np.log(path/strike) + (r + 0.5 * np.power(sigma, 2)) * ttm_vec )/(sigma * np.power(ttm_vec, 0.5))
    # interesting, the first line produces a TypeError: only size-1 arrays can be converted to Python scalars
    # maybe it doesn't affect np
    bs_delta = stats.norm.cdf(d1, 0, 1)
    return bs_delta


#%%
def ttm_vector(ttm, numSteps, numPaths):
    ttm_vec = np.zeros(numSteps)
    ttm_vec[0] = ttm
    dt = ttm/numSteps
    
    for i in range(1, numSteps):
        
        if i == numSteps-1:  
            ttm_vec[i] = 0.00000001
            # ttm_vec[i] = ttm_vec[i-1] - dt + 0.00000001
        else:
            ttm_vec[i] = ttm_vec[i-1] - dt
            
    ttm_vec = ttm_vec[:, np.newaxis]
    ttm_vec = np.tile(ttm_vec, (1, numPaths))
    return ttm_vec
#%%
def delta_hedge(S0, strike, sigma, r, ttm, numSteps, numPaths, quantity):
    
    dt =ttm/numSteps
    stock_pos = np.zeros((numSteps, numPaths))
    int_cost = np.zeros((numSteps, numPaths))
    stock_cuml = np.zeros((numSteps, numPaths))
    
    path = simpath(S0, r, sigma, ttm, numSteps, numPaths)
    ttm_vec = ttm_vector(ttm, numSteps, numPaths)
    delta = bs_delta(path, strike, sigma, ttm_vec, r, numPaths)
    delta_chg = np.diff(delta, axis=0)
    stock_pos[0, :] = delta[0, :] * path[0, :] * quantity
    stock_pos[1:, :] = np.multiply( delta_chg[:, :], path[1:, :]) * quantity
    stock_cuml[0, :] = stock_pos[0, :]
    
    
    for i in range(0, numSteps):
        if i < 1:
            int_cost[i, :] = 0
            stock_cuml[i, :] = stock_pos[0, :]
        else:
            int_cost[i, :] = stock_cuml[i-1, :]  * (np.exp(r * dt) - 1)
            stock_cuml[i, :] = stock_cuml[i-1, :] + stock_pos[i, :] + int_cost[i, :]
    
    cpth = []
    cdel = []
    cstockpos = []
    cstockcuml = []
    cintcost = []
    cttm_vec = []
        
    for i in range(0, numPaths):
        cpth.append('path_' + str(i))
        cdel.append('delta_' + str(i))
        cstockpos.append('stockpos_' + str(i))
        cstockcuml.append('stockcuml_' + str(i))
        cintcost.append('intcost_' + str(i))
        cttm_vec.append('ttmvec_' + str(i))
            
    path = pd.DataFrame(path, columns = cpth, index=range(0, numSteps))
    delta = pd.DataFrame(delta, columns = cdel, index=range(0, numSteps))
    stock_pos = pd.DataFrame(stock_pos, columns = cstockpos, index=range(0, numSteps))
    stock_cuml = pd.DataFrame(stock_cuml, columns = cstockcuml, index=range(0, numSteps))
    int_cost = pd.DataFrame(int_cost, columns = cintcost, index=range(0, numSteps))
    ttm_vec_ = pd.DataFrame(ttm_vec, columns = cttm_vec, index=range(0, numSteps))
   
    hedge_strat = pd.concat([path, delta, stock_pos, stock_cuml, int_cost, ttm_vec_], axis = 1)
    hedge_strat.columns = hedge_strat.columns.str.split('_', expand=True)
    pd.set_option('display.precision', 2) 
            
    return hedge_strat
#%%
def hedge_error(call0, df, strike, sigma, ttm, r, numSteps, numPaths, quantity):
    
    # call_0 = bs_call(S0, strike, sigma, ttm, r)
    
    #1. establish the moneyness of the position
    # stock_end = df.iloc[-1:, 0: numPaths] # this creates a multi-index, so need to drop a level
    # stock_end.columns = stock_end.columns.droplevel(level=0). but the following is easier:
    stock_end = df['path'][-1:]
    stock_cuml = df['stockcuml'][-1:] 
    
    # create the df that holds cashflows_end
    cashflow_end = np.zeros((1, numPaths))    
    
    for i in range(0, numPaths):
        if stock_end.iloc[0, i] > strike:
            cashflow_end[:, i] = strike * quantity - stock_cuml.iloc[:, i]   
        else:
            cashflow_end[:, i] = - stock_cuml.iloc[:, i]   
    
    hedge_results = call0 - np.absolute(cashflow_end) * np.exp(-r * ttm/numSteps)/quantity
    
    return [hedge_results, cashflow_end]
 #%%
S0 = 100
strike = 100
# t = dt.datetime(2022, 1, 1)
# M = dt.datetime(2022, 6, 1)
ttm = 1
r = 0.05
sigma = 0.50
numSteps = 5
numPaths = 1
mu = 0
# ttm = (M - t).days/365
epsilon = 0.02
quantity = 10000
call0 = bs_call(S0, strike, sigma, ttm, r)
hedge_strat = delta_hedge(S0, strike, sigma, r, ttm, numSteps, numPaths, quantity)
path = simpath(S0, r, sigma, ttm, numSteps, numPaths)
hedge_results = hedge_error(call0, hedge_strat, strike, sigma, ttm, r, numSteps, numPaths, quantity)

# #%%
# for i in range(1, numSteps-1):
#     print(i)
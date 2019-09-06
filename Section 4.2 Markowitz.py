#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: heshan
"""

######################### Section 4.2 #############################
# Reference: https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f

import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
import scipy.optimize as sco
import time

path = '/Users/heshan/Desktop/Portfolio Optimization/'
os.chdir(path)

###################### Data import and cleansing #######################
# Import data
data_aapl = pd.read_csv('AAPL.csv')
data_goog = pd.read_csv('GOOG.csv')
data_tsla = pd.read_csv('TSLA.csv')
data_ms = pd.read_csv('MS.csv')
data_market = pd.read_csv('SP500.csv')

datadict = {'AAPL':data_aapl, 'GOOG':data_goog, 'TSLA':data_tsla, 'MS':data_ms, 'Market':data_market}
keylist = ['AAPL','GOOG','TSLA','MS','Market']

origin_data = pd.DataFrame(index=pd.to_datetime(data_aapl['Date']), columns=keylist)
for key in keylist:
    data = datadict[key]
    data.index = pd.to_datetime(data['Date'])
    origin_data[key] = data['Adj Close']
prices_all = origin_data.copy()

prices = prices_all.iloc[:,:4]
prices.head()

# Calculate returns from prices
returns = prices.pct_change()
returns.dropna(inplace=True)
returns.head()

# Define some parameters
train_window = 63
alpha = 0.89

returns = returns.iloc[-train_window:,]

#################### # Efficient Frontier and Market Portfolio ################

# Function to evaluate portfolio performance
def portfolio_daily_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *1
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(1)
    return std, returns

# Function to generate random portfolios
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_daily_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

# Calculate mean and covariance
mean_returns = returns.ewm(alpha=alpha,min_periods=train_window).mean().iloc[-1,]
cov_matrix = returns.ewm(alpha=alpha,min_periods=train_window).cov().iloc[-4:,]

# Number of random portfolios
num_portfolios = 2500

# Assumption on annual risk-free rate
risk_free_rate = 0.01


# Calculate portfolio volatility
def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_daily_performance(weights, mean_returns, cov_matrix)[0]

# Get min variance
def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    # All in constraint
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Long only constraint
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    # Minimize variance
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

# Minimize portfolio volatility
def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_daily_performance(weights, mean_returns, cov_matrix)[1]
    # All in constraint
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Long only constraint
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', \
                          bounds=bounds, constraints=constraints)
    return result

# Get efficient frontier
def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_daily_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

# Optimizer to maximize sharpe ratio
def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    # All in constraint
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Long only constraint
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)

# Function to show the efficient frontier and random portfolios
def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    # Generate random portfolios
    results, _ = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    print (results)
    
    # Find the maximum SR portfolio
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_daily_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=returns.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    # Find the minimum variance portfolio
    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_daily_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=returns.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    # Print results of maximum SR portfolio & minimum variance portfolio
    print ("-"*80)
    print ("Maximum Sharpe Ratio Portfolio Allocation\n")
    print ("Annualised Return:", round(rp,2))
    print ("Annualised Volatility:", round(sdp,2))
    print ("\n")
    print (max_sharpe_allocation)
    print ("-"*80)
    print ("Minimum Volatility Portfolio Allocation\n")
    print ("Annualised Return:", round(rp_min,2))
    print ("Annualised Volatility:", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)
    
    # Plot the efficient frontier with market portfolio
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Market Portfolio')
    plt.plot([0, sdp, 2*sdp], [risk_free_rate, rp, risk_free_rate+2*(rp-risk_free_rate)], color='k', \
             linestyle='-', linewidth=2)
    target = np.linspace(rp_min, 0.024, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', \
             label='Efficient Frontier')
    plt.suptitle('Market portfolio and efficient frontier in the Markowitz Portfolio')
    plt.xlabel('Daily Risk (Standard Deviation)')
    plt.ylabel('Expected Daily Returns')
    plt.legend(labelspacing=0.8, loc='upper left')
    axes = plt.gca()
    axes.set_ylim([0.002,0.03])
    fig.savefig("Morkowitz_Example.pdf", dpi=600)
    
# Count eclipse time
start_time = time.perf_counter()

# Format of figures
SMALL_SIZE = 14
MEDIUM_SIZE = 17
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

end_time = time.perf_counter()
print (end_time - start_time)

################ # Markowitz Protfolio Allocation Example #################
def neg_return(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_daily_performance(weights, mean_returns, cov_matrix)
    return (-p_ret)

# Optimizor to maximize returns
def max_return(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    # All in constraint
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Long only constraint
    bound = (0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_return, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

returns_all = prices_all.pct_change()

alpha = 0.89
returns = prices.pct_change()
# Markowitz Portolio Optimization
markowitz = returns.copy().loc[pd.to_datetime('2017-01-01'):,]
rf = (1+0.01)**(1/365)-1
train_window = 63
N = 4
for i in range(N):
    markowitz[markowitz.columns[i]+' Weight'] = 0

T = len(markowitz.index)
markowitz_SR = markowitz.copy()
markowitz.head(5)

start_time = time.process_time()

# Rolling train and test data
# Use 63-day data to allocate weight for next day
# Use one-year exp smoothing return to estimate future one-day return, smoothing alpha=0.89
# Use last 63-day sd to estimate future one-day sd

for t in range(T-train_window-1):
    train_index = markowitz.index[t:t+train_window]
    train_data = markowitz.loc[train_index,keylist[0:N]]
    mean_returns = train_data.ewm(alpha=alpha,min_periods=train_window).mean().iloc[-1,]
    cov_matrix = train_data.ewm(alpha=alpha,min_periods=train_window).cov().iloc[-4:,]

    for i in range(N):
        # Maximize Return
        markowitz.loc[markowitz.index[t+train_window+1],markowitz.columns[i]+' Weight'] = \
            max_return(mean_returns, cov_matrix, rf).x[i]
        markowitz_SR.loc[markowitz.index[t+train_window+1],markowitz.columns[i]+' Weight'] = \
            max_sharpe_ratio(mean_returns, cov_matrix, rf).x[i]
        
end_time = time.process_time()
print (end_time-start_time)

# Markowitz porfolio to maximize returns
drop_window = train_window

markowitz = markowitz.iloc[drop_window+1:,].copy()
markowitz['Daily return'] = 0
for i in range(N):
    markowitz['Daily return'] += markowitz.iloc[:,i]*markowitz.loc[:,markowitz.columns[i]+' Weight']
markowitz['Cum return'] = (1+markowitz['Daily return']).cumprod()
markowitz.tail(5)

# Markowitz porfolio to the Sharpe ratio
markowitz_SR = markowitz_SR.iloc[drop_window+1:,].copy()
markowitz_SR['Daily return'] = 0
for i in range(N):
    markowitz_SR['Daily return'] += markowitz_SR.iloc[:,i]*markowitz_SR.loc[:,markowitz_SR.columns[i]+' Weight']
markowitz_SR['Cum return'] = (1+markowitz_SR['Daily return']).cumprod()
markowitz_SR.tail(5)

markowitz.to_csv('markowitz_return.csv')
markowitz_SR.to_csv('markowitz_SR.csv')

# Plot cumulative returns of two Markowitz portfolios
fig = plt.figure(figsize=(14, 7))
plt.plot(markowitz['Cum return'].index, markowitz['Cum return'], lw=3, alpha=0.8)
plt.plot(markowitz_SR['Cum return'].index, markowitz_SR['Cum return'], lw=3, alpha=0.8)
plt.plot(markowitz['Cum return'].index, (1+returns_all.loc[markowitz['Cum return'].index,'Market']).cumprod(), \
         lw=3, alpha=0.8)
plt.legend(labels = ('Markowitz Portfolio (Max Return)',\
                     'Markowitz Portfolio (Max Sharpe Ratio)',\
                     'S&P 500'),loc='upper left')
plt.ylabel('Cumulative return')
plt.suptitle('Markowitz Potfolio Optimization')
fig.savefig("Morkowitz_Example2.pdf")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: heshan
"""

##################### Section 6 & Figures for Section 5.4 #####################

# Import libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set path
path = '/Users/heshan/Desktop/Portfolio Optimization/'
os.chdir(path)

# Data import and cleansing
data_aapl = pd.read_csv('AAPL.csv')
data_goog = pd.read_csv('GOOG.csv')
data_market = pd.read_csv('SP500.csv')
datadict = {'AAPL':data_aapl, 'GOOG':data_goog, 'Market':data_market}
keylist = ['AAPL','GOOG', 'Market']
origin_data = pd.DataFrame(index=pd.to_datetime(data_market['Date']),\
                           columns=keylist)
# Only keep adjusted closed prices and dates
for key in keylist:
    data = datadict[key]
    data.index = pd.to_datetime(data['Date'])
    origin_data[key] = data['Adj Close']
prices = origin_data.copy()

prices.tail()

# Plot font settings
# reference: 
# https://stackoverflow.com/questions/3899980/
# how-to-change-the-font-size-on-a-matplotlib-plot

SMALL_SIZE = 16
MEDIUM_SIZE = 19
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE) # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE) # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE) # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE) # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE) # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE) # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title

################## Load some results from our models ##################

### Markowitz portfolio

# Set first 70% data as training data
training_data_index = int(len(prices.index)*0.7)
data = prices.copy().iloc[training_data_index:,]

markowitz = pd.read_csv('markowitz.csv', index_col='Date')
markowitz.index = pd.to_datetime(markowitz.index)

markowitz.head()

### DQN (Returns)

# Load results
repeat_times = 20
os.chdir(path+'Data Store/Q Test Return')
#os.chdir(path+'Data Store/Q Test')

# Load results to calculate mean and variance
template = pd.read_csv('DQN_test0.csv', index_col='Date')
DQN_returns_pred_test = pd.DataFrame(index=template.index)
DQN_returns_pred_test.index = pd.to_datetime(DQN_returns_pred_test.index)

for iterations in range(repeat_times):
    i = pd.read_csv('DQN_test'+str(iterations)+'.csv', index_col='Date')
    DQN_returns_pred_test['Portfolio Value '+str(iterations)] = i['Portfolio Value']
os.chdir(path)

# Mean and std for cumulative returns
DQN_returns_pred_test['Mean'] = DQN_returns_pred_test.mean(axis = 1)
DQN_returns_pred_test['Std'] = DQN_returns_pred_test.std(axis = 1)

# Equal allocation benchmark
DQN_returns_pred_test['AAPL Daily Return'] = data.pct_change()['AAPL']
DQN_returns_pred_test['GOOG Daily Return'] = data.pct_change()['GOOG']
DQN_returns_pred_test.loc[DQN_returns_pred_test.index[0],'AAPL Daily Return'] = 0
DQN_returns_pred_test.loc[DQN_returns_pred_test.index[0],'GOOG Daily Return'] = 0
DQN_returns_pred_test['Benchmark'] = (1+0.5*(DQN_returns_pred_test['AAPL Daily Return']+\
                                            DQN_returns_pred_test['GOOG Daily Return'])).cumprod()

DQN_returns_pred_test.tail()

DQN_returns_pred_test_return = DQN_returns_pred_test.copy()

### DQN (Prices)

# Load results
repeat_times = 20
#os.chdir(path+'Data Store/Q Test Return')
os.chdir(path+'Data Store/Q Test')

# Load results to calculate mean and variance
DQN_returns_pred_test = pd.DataFrame(index=template.index)
DQN_returns_pred_test.index = pd.to_datetime(DQN_returns_pred_test.index)

for iterations in range(repeat_times):
    i = pd.read_csv('DQN_test'+str(iterations)+'.csv', index_col='Date')
    DQN_returns_pred_test['Portfolio Value '+str(iterations)] = i['Portfolio Value']
os.chdir(path)

# Mean and std for cumulative returns
DQN_returns_pred_test['Mean'] = DQN_returns_pred_test.mean(axis = 1)
DQN_returns_pred_test['Std'] = DQN_returns_pred_test.std(axis = 1)

# Equal allocation benchmark
DQN_returns_pred_test['AAPL Daily Return'] = data.pct_change()['AAPL']
DQN_returns_pred_test['GOOG Daily Return'] = data.pct_change()['GOOG']
DQN_returns_pred_test.loc[DQN_returns_pred_test.index[0],'AAPL Daily Return'] = 0
DQN_returns_pred_test.loc[DQN_returns_pred_test.index[0],'GOOG Daily Return'] = 0
DQN_returns_pred_test['Benchmark'] = (1+0.5*(DQN_returns_pred_test['AAPL Daily Return']+\
                                            DQN_returns_pred_test['GOOG Daily Return'])).cumprod()

DQN_returns_pred_test.tail()

DQN_returns_pred_test_prices = DQN_returns_pred_test.copy()

### DGN (Returns)

# Load results
repeat_times = 20
os.chdir(path+'Data Store/G Test Return')
#os.chdir(path+'Data Store/G Test')

DGN_returns_pred_test = pd.DataFrame(index=template.index)
DGN_returns_pred_test.index = pd.to_datetime(DGN_returns_pred_test.index)

for iterations in range(repeat_times):
    i = pd.read_csv('DGN_test'+str(iterations)+'.csv', index_col='Date')
    DGN_returns_pred_test['Portfolio Value '+str(iterations)] = i['Portfolio Value']
os.chdir(path)

# Calculate mean and std cumulative returns
DGN_returns_pred_test['Mean'] = DGN_returns_pred_test.mean(axis = 1)
DGN_returns_pred_test['Std'] = DGN_returns_pred_test.std(axis = 1)

# Equal allocation benchmark
DGN_returns_pred_test['AAPL Daily Return'] = data.pct_change()['AAPL']
DGN_returns_pred_test['GOOG Daily Return'] = data.pct_change()['GOOG']
DGN_returns_pred_test.loc[DGN_returns_pred_test.index[0],'AAPL Daily Return'] = 0
DGN_returns_pred_test.loc[DGN_returns_pred_test.index[0],'GOOG Daily Return'] = 0
DGN_returns_pred_test['Benchmark'] = (1+0.5*(DGN_returns_pred_test['AAPL Daily Return']+\
                                            DGN_returns_pred_test['GOOG Daily Return'])).cumprod()

DGN_returns_pred_test.tail()

DGN_returns_pred_test_return = DGN_returns_pred_test.copy()

### DGN (Prices)

# Load results
repeat_times = 20
#os.chdir(path+'Data Store/G Test Return')
os.chdir(path+'Data Store/G Test')

DGN_returns_pred_test = pd.DataFrame(index=template.index)
DGN_returns_pred_test.index = pd.to_datetime(DGN_returns_pred_test.index)

for iterations in range(repeat_times):
    i = pd.read_csv('DGN_test'+str(iterations)+'.csv', index_col='Date')
    DGN_returns_pred_test['Portfolio Value '+str(iterations)] = i['Portfolio Value']
os.chdir(path)

# Calculate mean and std cumulative returns
DGN_returns_pred_test['Mean'] = DGN_returns_pred_test.mean(axis = 1)
DGN_returns_pred_test['Std'] = DGN_returns_pred_test.std(axis = 1)

# Equal allocation benchmark
DGN_returns_pred_test['AAPL Daily Return'] = data.pct_change()['AAPL']
DGN_returns_pred_test['GOOG Daily Return'] = data.pct_change()['GOOG']
DGN_returns_pred_test.loc[DGN_returns_pred_test.index[0],'AAPL Daily Return'] = 0
DGN_returns_pred_test.loc[DGN_returns_pred_test.index[0],'GOOG Daily Return'] = 0
DGN_returns_pred_test['Benchmark'] = (1+0.5*(DGN_returns_pred_test['AAPL Daily Return']+\
                                            DGN_returns_pred_test['GOOG Daily Return'])).cumprod()

DGN_returns_pred_test.tail()

DGN_returns_pred_test_prices = DGN_returns_pred_test.copy()

### Continuous G-learning with Gaussian priors and linear dynamics

ContinuousG_returns = pd.read_csv('Continuous G-learning.csv', index_col='Date')
ContinuousG_returns.index = pd.to_datetime(ContinuousG_returns.index)
ContinuousG_returns.drop(index=ContinuousG_returns.index\
                         [ContinuousG_returns.index < DGN_returns_pred_test.index[0]], inplace=True)
ContinuousG_returns['Portfolio Value'] = ContinuousG_returns['Portfolio Value']/\
                         ContinuousG_returns.loc[ContinuousG_returns.index[0],'Portfolio Value']
ContinuousG_returns.head()

ContinuousG_returns.tail()



################## Figures for Section 5.4 (G-learning with Linear Dynamics) ##################

# Plot the cumulative returns of G-learning with Gaussian Priors and Linear Dynamics vs benchamrks
fig = plt.figure(figsize=(14, 7))
plt.plot(ContinuousG_returns['Portfolio Value'], lw=3, alpha=0.8, color = 'blue')
plt.plot(markowitz.loc[DGN_returns_pred_test.index,'Cum return']/\
         markowitz.loc[DGN_returns_pred_test.index[0],'Cum return'],\
         lw=2, alpha=0.8,label='linear',linestyle = '--', color = 'forestgreen')
plt.plot(DGN_returns_pred_test.loc[DGN_returns_pred_test.index,'Benchmark'],\
         linestyle = '--', lw=2, alpha=0.8, color = 'darkred')
plt.legend(labels = ('G-learning with Gaussian Priors',
                     'Markowitz (Max Return)','Equal Allocation'),loc='upper left',\
                     borderaxespad=0., fontsize=15.5)
plt.ylabel('Cumulative return')
axes = plt.gca()
axes.set_ylim([0.8,1.5])
plt.suptitle('Cumulative Returns of G-learning with Gaussian Priors (Testing Data)')
fig.savefig("Continuous_G_Benchmarks.pdf")

# Plot the allocation weights of G-learning with Gaussian Priors and Linear Dynamics
fig = plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(ContinuousG_returns['w0'])
plt.ylabel('Weights of AAPL')
plt.xticks(rotation=30)
fig.savefig("DGN_Benchmarks_Test.pdf")
plt.subplot(1, 2, 2)
sns.distplot(ContinuousG_returns['w0'], kde=False)
plt.ylabel('Frequency')
plt.xlabel('Weights of AAPL')
plt.suptitle('G-learning with Gaussian Priors Allocation Weights of AAPL (Testing Data)')
fig.savefig("Continuous_G_Benchmarks_Weights.pdf")

# Get daily returns from cumlative returns
def to_daily_return (data):
    dailt_return=data/data.shift(1)-1
    dailt_return.dropna(inplace=True)
    return (dailt_return)

# Returns, SR, Volitility and max draw down. 
def evaluate_performance(data, rf):
    performance_data = pd.DataFrame(data)
    performance_data.rename(columns = {performance_data.columns[0]:\
                                       'Daily Return'}, inplace=True)
    performance_data.loc[performance_data.index[1:],\
        'Cumulative Return'] = (1 +\
        performance_data.loc[performance_data.index[1:],\
        'Daily Return']).cumprod()
    performance_data.loc[performance_data.index[0],\
                         'Cumulative Return'] = 1

    # end of max drawdown period
    end=(np.maximum.accumulate(performance_data['Cumulative Return'])\
         - performance_data['Cumulative Return']).idxmax()
    # start of max drawdown period
    start = (performance_data['Cumulative Return'][:end]).idxmax()
    # Maximum Drawdown
    loss = performance_data['Cumulative Return'][end] / \
           performance_data['Cumulative Return'][start]-1
    
    # Return, volatility, Sharpe ratio
    annual_ret = performance_data['Daily Return'].mean()*252
    annual_risk = performance_data['Daily Return'].std()*np.sqrt(252)
    annual_sr = (annual_ret-rf)/annual_risk

    print (annual_ret, annual_risk, annual_sr, start, end, loss)

# Financial indicators of the G-learning with Gaussian priors method
rf = 0.01
evaluate_performance(to_daily_return(ContinuousG_returns['Portfolio Value']), rf)
evaluate_performance(to_daily_return(DGN_returns_pred_test['Benchmark']), rf)
evaluate_performance(to_daily_return(markowitz.loc[DGN_returns_pred_test.index,'Cum return']/\
         markowitz.loc[DGN_returns_pred_test.index[0],'Cum return']), rf)



################## Section 6.1 Comparision of Different Methods ##################

fig = plt.figure(figsize=(14, 7))
plt.plot(DQN_returns_pred_test['Mean'], lw=3, alpha=0.8, color = 'blue')
plt.plot(DGN_returns_pred_test['Mean'],linestyle = '-', lw=2, alpha=0.8,color='orange')
plt.plot(ContinuousG_returns['Portfolio Value'],linestyle = '-', lw=2, alpha=0.8, color='k')
plt.plot(markowitz.loc[DGN_returns_pred_test.index,'Cum return']/\
         markowitz.loc[DGN_returns_pred_test.index[0],'Cum return'],\
         lw=2, alpha=0.8,label='linear',linestyle = '--', color = 'forestgreen')
plt.plot(DGN_returns_pred_test.loc[DGN_returns_pred_test.index,'Benchmark'],\
         linestyle = '--', lw=2, alpha=0.8, color = 'darkred')
plt.legend(labels = ('DQN Mean','DGN Mean','G-learning with Gaussian Priors',\
                     'Markowitz (Max Return)','Equal Allocation'),loc='upper left',\
                     borderaxespad=0., fontsize=15.5)
plt.ylabel('Cumulative return')
axes = plt.gca()
axes.set_ylim([0.8,1.5])
plt.suptitle('Cumulative Returns of Different Methods (Testing Data)')
fig.savefig("Comparision_1.pdf")

fig = plt.figure(figsize=(14, 7))
plt.plot(DQN_returns_pred_test['Mean'], lw=3, alpha=0.8, color = 'blue')
plt.plot(DQN_returns_pred_test['Mean']+DQN_returns_pred_test['Std'],linestyle = ':', lw=2, alpha=0.8, color = 'blue')
plt.plot(DQN_returns_pred_test['Mean']-DQN_returns_pred_test['Std'],linestyle = ':', lw=2, alpha=0.8, color = 'blue')
plt.fill_between(DQN_returns_pred_test.index, DQN_returns_pred_test['Mean']-DQN_returns_pred_test['Std'],\
                 DQN_returns_pred_test['Mean']+DQN_returns_pred_test['Std'], color = 'lightsteelblue', alpha=0.5)
plt.plot(DGN_returns_pred_test['Mean'],linestyle = '-', lw=2, alpha=0.8, color='orange')
plt.plot(DGN_returns_pred_test['Mean']+DGN_returns_pred_test['Std'],linestyle = ':', lw=2, alpha=0.8, color = 'orange')
plt.plot(DGN_returns_pred_test['Mean']-DGN_returns_pred_test['Std'],linestyle = ':', lw=2, alpha=0.8, color = 'orange')
plt.fill_between(DGN_returns_pred_test.index, DGN_returns_pred_test['Mean']-DGN_returns_pred_test['Std'],\
                 DGN_returns_pred_test['Mean']+DGN_returns_pred_test['Std'], color = 'bisque', alpha=0.5)
plt.legend(labels = ('DQN Mean','DQN Mean-Std','DQN Mean+Std','DGN Mean',\
                     'DGN Mean-Std','DGN Mean+Std'),loc='upper left',\
                     borderaxespad=0., fontsize=15.5)
plt.ylabel('Cumulative return')
#axes = plt.gca()
#axes.set_ylim([0.8,1.6])
plt.suptitle('Cumulative Returns of DQN and DGN with Errors')
fig.savefig("Comparision_2.pdf")



################## Section 6.2 Compare Returns and Prices in State Space ##################

# Plot the cumulative returns of DQN (Returns), DGN (Returns), DQN (Prices), DGN (Prices)
fig = plt.figure(figsize=(14, 7))
plt.plot(DQN_returns_pred_test_prices['Mean'], lw=2, alpha=1, color = 'blue')
plt.plot(DGN_returns_pred_test_prices['Mean'],linestyle = '-', lw=2, alpha=0.8,color='orange')
plt.plot(DQN_returns_pred_test_return['Mean'], lw=2, alpha=1, linestyle = '-.', color = 'green')
plt.plot(DGN_returns_pred_test_return['Mean'],linestyle = '-.', lw=2, alpha=0.8,color='red')
plt.legend(labels = ('DQN (prices)','DGN (prices)','DQN (returns)','DGN (returns)'),\
                     borderaxespad=0., fontsize=15.5)
plt.ylabel('Cumulative return')
plt.suptitle('Cumulative Returns of DQN and DGN with different State Spaces')
fig.savefig("Comparision_3.pdf")

# Plot the cumulative returns with errors of DQN (Returns), DQN (Prices)
fig = plt.figure(figsize=(14, 7))
plt.plot(DQN_returns_pred_test_prices['Mean'], lw=2, alpha=1, color = 'blue')
plt.plot(DQN_returns_pred_test_prices['Mean']+DQN_returns_pred_test_prices['Std'],linestyle = ':', \
         lw=2, alpha=0.8, color = 'blue')
plt.plot(DQN_returns_pred_test_prices['Mean']-DQN_returns_pred_test_prices['Std'],linestyle = ':', \
         lw=2, alpha=0.8, color = 'blue')
plt.fill_between(DQN_returns_pred_test_prices.index, DQN_returns_pred_test_prices['Mean']-\
                 DQN_returns_pred_test_prices['Std'],\
                 DQN_returns_pred_test_prices['Mean']+DQN_returns_pred_test_prices['Std'], \
                 color = 'lightsteelblue', alpha=0.5)
plt.plot(DQN_returns_pred_test_return['Mean'],lw=2, linestyle = '-.', color='green')
plt.plot(DQN_returns_pred_test_return['Mean']+DQN_returns_pred_test_return['Std'],linestyle = ':', \
         lw=2, alpha=0.8, color = 'green')
plt.plot(DQN_returns_pred_test_return['Mean']-DQN_returns_pred_test_return['Std'],linestyle = ':', \
         lw=2, alpha=0.8, color = 'green')
plt.fill_between(DQN_returns_pred_test_return.index, DQN_returns_pred_test_return['Mean']-\
                 DQN_returns_pred_test_return['Std'],\
                 DQN_returns_pred_test_return['Mean']+DQN_returns_pred_test_return['Std'], \
                 color = 'honeydew', alpha=0.5)
plt.legend(labels = ('DQN (Prices) Mean','DQN (Prices) Mean-Std','DQN (Prices) Mean+Std','DQN (Returns) Mean',\
                     'DQN (Returns) Mean - Std','DQN (Returns) Mean + Std'),loc='upper left',\
                     borderaxespad=0., fontsize=15.5)
axes = plt.gca()
axes.set_ylim([0.8,1.6])
plt.ylabel('Cumulative return')
plt.suptitle('Cumulative Returns of DQN with different State Spaces')
fig.savefig("Comparision_4.pdf")

# Plot the cumulative returns with errors of DGN (Returns), DGN (Prices)
fig = plt.figure(figsize=(14, 7))
plt.plot(DGN_returns_pred_test_prices['Mean'], lw=2, alpha=1, color = 'orange')
plt.plot(DGN_returns_pred_test_prices['Mean']+DGN_returns_pred_test_prices['Std'],linestyle = ':', lw=2, \
         alpha=0.8, color = 'orange')
plt.plot(DGN_returns_pred_test_prices['Mean']-DGN_returns_pred_test_prices['Std'],linestyle = ':', lw=2, \
         alpha=0.8, color = 'orange')
plt.fill_between(DGN_returns_pred_test_prices.index, DGN_returns_pred_test_prices['Mean']-\
                 DGN_returns_pred_test_prices['Std'], DGN_returns_pred_test_prices['Mean']+\
                 DGN_returns_pred_test_prices['Std'], color = 'lemonchiffon', alpha=0.5)
plt.plot(DGN_returns_pred_test_return['Mean'],lw=2, linestyle = '-.', color='red')
plt.plot(DGN_returns_pred_test_return['Mean']+DGN_returns_pred_test_return['Std'],linestyle = ':', lw=2, \
         alpha=0.8, color = 'red')
plt.plot(DGN_returns_pred_test_return['Mean']-DGN_returns_pred_test_return['Std'],linestyle = ':', lw=2, \
         alpha=0.8, color = 'red')
plt.fill_between(DGN_returns_pred_test_return.index, DGN_returns_pred_test_return['Mean']-\
                 DGN_returns_pred_test_return['Std'],\
                 DGN_returns_pred_test_return['Mean']+DGN_returns_pred_test_return['Std'], color = 'mistyrose', \
                 alpha=0.5)
plt.legend(labels = ('DGN (Prices) Mean','DGN (Prices) Mean - Std','DGN (Prices) Mean + Std','DGN (Returns) Mean',\
                     'DGN (Returns) Mean - Std','DGN (Returns) Mean + Std'),loc='upper left',\
                     borderaxespad=0., fontsize=15.5)
axes = plt.gca()
axes.set_ylim([0.8,1.5])
plt.ylabel('Cumulative return')
plt.suptitle('Cumulative Returns of DGN with different State Spaces')
fig.savefig("Comparision_5.pdf")






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: heshan
"""
##################### Section 5.4 (Modelling only)#####################

#Codes for figure are in the file 'Section 6.ipynb'

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

# Import price data
data_aapl = pd.read_csv('AAPL.csv')
data_goog = pd.read_csv('GOOG.csv')
data_market = pd.read_csv('SP500.csv')

datadict = {'AAPL':data_aapl, 'GOOG':data_goog,'Market':data_market}
keylist = ['AAPL','GOOG','Market']
origin_data = pd.DataFrame(index=pd.to_datetime(data_market['Date']),\
                           columns=keylist)
for key in keylist:
    data = datadict[key]
    data.index = pd.to_datetime(data['Date'])
    origin_data[key] = data['Adj Close']
prices = origin_data.copy()

# Calculate returns
returns = prices.pct_change()
returns.dropna(inplace=True)
returns_all = returns.copy()
returns.head()

########### Section 5.4 Function of G-learning with linear dynamics ###########

# Get beta in CAPM model
def get_beta(Y,X):
    X = sm.add_constant(X)
    model = sm.OLS(Y,X).fit()
    return model.params[1]

# Function to run the G learning with linear dynamics
def G_learning_with_linear_dynamics(start, end, data, rf, gamma, beta,\
                                    iterations):
    # Locate data
    returns = data.iloc[start:end,]
    T = len(returns.index)
    
    # Set some constants for convenience
    I = np.identity(N)
    In = np.concatenate((I,-I))
    
    # Initialize A_0, A_1, \Sigma, \bar{X}, \bar{a}
    A0t = []
    A1t = []
    sigmaat = []

    xbar = []
    abar = []
    dxt = []
    dat = []

    for t in range(T):
        A0t.append(np.zeros(shape=(2*N,1)))
        A1t.append(np.zeros(shape=(2*N,N)))
        sigmaat.append(np.diag(np.array([1]*(2*N))))

        xbar.append(np.array([[1/N]]*N))
        abar.append(np.array(A0t[0]+np.dot(A1t[0],xbar[t])))
        dxt.append(np.array([[0]]*N))
        dat.append(np.array([[0]]*(2*N)))
    
    # Terminal condition
    duT = -xbar[T-2]-np.dot(In.T,abar[T-1])
    dat[T-1]=np.concatenate([np.maximum(duT,np.array([[0]]*N)),\
                             np.maximum(-duT,np.array([[0]]*N))])
    
    # Initialize market signal z, returns r
    z = np.array([returns['Market']-rf]*N).T
    r = np.array(returns[keylist[0:N]]-rf)
    zt = []
    rt = []
    for t in range(T):
        zt.append((z[t,:].reshape((N,1))))
        rt.append((r[t,:].reshape((N,1))))
    
    # Get beta in CAPM model
    beta_list = np.array([])
    for key in keylist[0:N]:
        beta_list = np.append(beta_list,\
                              get_beta(data.iloc[start:end,][key],\
                              data.iloc[start:end,]['Market']))
    # Make assumptions on W, M
    W = np.diag(beta_list)
    M = np.identity(N)*1e-10
    
    # Error epsilon of CAPM model
    epsilon = r.copy()
    for i in range(N):
        epsilon[:,i] = r[:,i] - beta_list[i]*z[:,i]
    sigmat = np.cov(epsilon.T)
    
    # Further assumptions on the penalty parameters
    lambd = 1
    thetap = np.array([[0.1]]*N)
    thetan = np.array([[0.1]]*N)
    psi = np.array([[0.1]]*N)
    kp = np.array([[0.0]])
    kn = np.array([[0.0]])
    
    # Parameters related to the penalized return
    Raat_tilda = []
    Rxxt_tilda = []
    Raxt_tilda = []
    Rat_tilda = []
    Rxt_tilda = []
    for t in range(T):
        Raat_tilda.append\
            (np.concatenate((np.concatenate((-M-lambd*sigmat,\
                                             M+lambd*sigmat),axis=1),\
                             np.concatenate((M+lambd*sigmat,\
                                             -M-lambd*sigmat),axis=1))\
                            ,axis=0))
        Rxxt_tilda.append(-lambd*sigmat)
        Raxt_tilda.append(np.concatenate((-M-2*lambd*sigmat-thetap,\
                                          M+2*lambd*sigmat-thetan),\
                                         axis=0))
        Rat_tilda.append(np.concatenate((np.dot(W,zt[t])-\
                                         kp,np.dot(W,zt[t])-kn),\
                                        axis=0))
        Rxt_tilda.append(np.linalg.multi_dot([W-psi,zt[t]]))
    
    # Initilize parameters for parametrizations
    Dt = [0]*T
    Ht = [0]*T
    ft = [0]*T

    D2t = [0]*T
    H2t = [0]*T
    f2t = [0]*T

    At_hat = [0]*T
    Bt_hat = [0]*T
    Ct_hat = [0]*T

    Faat = [0]*T
    Fxxt = [0]*T
    Faxt = [0]*T
    Fat = [0]*T
    Fxt = [0]*T
    ft_hat = [0]*T

    Gaat = [0]*T
    Gxxt = [0]*T
    Gaxt = [0]*T
    Gat = [0]*T
    Gxt = [0]*T
    gt = [0]*T

    sigmaa_tilde = sigmaat.copy()
    
    # Run the loop many times
    for i in range(iterations): 

        #print (beta)

        # Parametrization for R
        Raat = Raat_tilda.copy()
        Rxxt = Rxxt_tilda.copy()
        Raxt = Raxt_tilda.copy()
        Rat = []
        Rxt = []
        Rt = []
        
        for t in range(T): # (cont.) Parametrization of R
            Rat.append(Rat_tilda[t]+2*np.dot(Raat_tilda[t],abar[t])+\
                       np.dot(Raxt_tilda[t],xbar[t]))
            Rxt.append(Rxt_tilda[t]+2*np.dot(Rxxt_tilda[t],xbar[t])+\
                       np.dot(Raxt_tilda[t].T,abar[t]))
            Rt.append(np.linalg.multi_dot([abar[t].T,Raat_tilda[t],\
                                           abar[t]])+\
                      np.linalg.multi_dot([xbar[t].T,Rxxt_tilda[t],\
                                           xbar[t]])+\
                      np.linalg.multi_dot([abar[t].T,Raxt_tilda[t],\
                                           xbar[t]])+\
                      np.dot(abar[t].T,Rat_tilda[t])+\
                      np.dot(xbar[t].T,Rxt_tilda[t]))
        
        # Parameters at time T
        Dt[T-1] = Rxxt[T-1]
        Ht[T-1] = Rxt[T-1]+np.dot(Raxt[T-1].T,dat[T-1])
        ft[T-1] = np.linalg.multi_dot([dat[T-1].T,Raat[T-1],\
                                       dat[T-1]])+\
                  np.dot(dat[T-1].T,Rat[T-1])+Rt[T-1]
        
        # Backward iteration to update parametrization of 
        # G, F, E[F_{t+1}]
        for t in range(T-2,-1,-1):

            At_hat[t] = np.diag(np.diag(I+rf*I+np.dot(W,zt[t])-\
                                    np.dot(M.T,np.dot(In.T,abar[t]))))
            Bt_hat[t] = At_hat[t] - np.multiply((xbar[t]+\
                                             np.dot(In.T,abar[t])),M)
            Ct_hat[t] = np.dot(At_hat[t],xbar[t]+np.dot(In.T,abar[t]))\
                        -xbar[t+1]

            # Parametrization for E[F_{t+1}]
            Faat[t] = np.linalg.multi_dot([In, Bt_hat[t].T, Dt[t+1],\
                                           Bt_hat[t], In.T])+\
                      np.linalg.multi_dot([In,np.multiply(Dt[t+1],\
                                                      sigmat),In.T])
            Fxxt[t] = np.linalg.multi_dot([At_hat[t].T, Dt[t+1], \
                                           At_hat[t]])+\
                      np.multiply(Dt[t+1],sigmat)
            Faxt[t] = 2*np.linalg.multi_dot([In, Bt_hat[t].T,Dt[t+1],\
                                             At_hat[t]])+\
                      2*np.linalg.multi_dot([In,np.multiply(Dt[t+1],\
                                                            sigmat)])
            Fat[t] = np.linalg.multi_dot([In, Bt_hat[t].T, Ht[t+1]])+\
                     2*np.linalg.multi_dot([In, Bt_hat[t].T, Dt[t+1],\
                                            Ct_hat[t]])+\
                     2*np.linalg.multi_dot([In,np.multiply(Dt[t+1],\
                                                           sigmat),\
                                    (xbar[t]+np.dot(In.T,abar[t]))])
            Fxt[t] = np.linalg.multi_dot([At_hat[t].T, Ht[t+1]])+\
                     2*np.linalg.multi_dot([At_hat[t].T, Dt[t+1], \
                                            Ct_hat[t]])+\
                     2*np.linalg.multi_dot([np.multiply(Dt[t+1],\
                                                        sigmat),\
                                       (xbar[t]+np.dot(In.T,abar[t]))])
            ft_hat[t] = ft[t+1]+np.dot(Ct_hat[t].T,Ht[t+1])+\
                np.linalg.multi_dot([Ct_hat[t].T, Dt[t+1],Ct_hat[t]])+\
                np.linalg.multi_dot([(xbar[t]+\
                                      np.dot(In.T,abar[t])).T,\
                                     np.multiply(Dt[t+1],sigmat),\
                                     (xbar[t]+np.dot(In.T,abar[t]))])

            # Parametrization for G_t
            Gaat[t] = Raat[t]+gamma*Faat[t]
            Gxxt[t] = Rxxt[t]+gamma*Fxxt[t]
            Gaxt[t] = Raxt[t]+gamma*Faxt[t]
            Gat[t] = Rat[t]+gamma*Fat[t]
            Gxt[t] = Rxt[t]+gamma*Fxt[t]
            gt[t] = Rt[t]+gamma*ft_hat[t]

            dat_tilde = A0t[t]+np.dot(A1t[t],xbar[t]) - abar[t]
            sigmaa_tilde[t] = np.linalg.inv(sigmaat[t])-2*beta*Gaat[t]
            
            # Eigenvalue method to slove the problem of 
            # matrix inconvertibility
            max_egvalue = np.linalg.eig(Gaat[t])[0][0]
            count = 0
            while (np.linalg.det(sigmaa_tilde[t])<=0):
                if (count==0):
                    beta = abs(1/max_egvalue)
                else:
                    beta = abs(beta/max_egvalue)

                if (beta<1e-50):
                    beta = 0           
                sigmaa_tilde[t] = np.linalg.inv(sigmaat[t])-\
                                  2*beta*Gaat[t]
                count +=1

            # Parametrization for F_t
            bt = abar[t]-A0t[t]-np.dot(A1t[t],xbar[t])
            
            if beta == 0: # Special case for beta is zero
                tau = np.diag(np.array([1]*(2*N)))
                gammab = np.diag(np.array([0]*(2*N)))
                Lb = 0
            else: 
                tau = 1/beta*(np.linalg.inv(sigmaat[t])-\
                   np.linalg.multi_dot([np.linalg.inv(sigmaat[t]).T,\
                                       np.linalg.inv(sigmaa_tilde[t]),\
                                       np.linalg.inv(sigmaat[t])]))
                gammab = np.dot(np.linalg.inv(sigmaa_tilde[t]),\
                                np.linalg.inv(sigmaat[t]))
                Lb=1/(2*beta)*(np.log(np.linalg.det(sigmaat[t]))+\
                               np.log(np.linalg.det(sigmaa_tilde[t])))

            Eax = np.dot(gammab,A1t[t])+\
                  0.5*beta*np.dot(np.linalg.inv(sigmaa_tilde[t]),\
                                  Gaxt[t])
            Dax = np.dot(Gaxt[t].T,gammab)-np.dot(A1t[t].T,tau)
            Ea = np.linalg.multi_dot([A1t[t].T,gammab,Gat[t]])+\
                 beta*np.linalg.multi_dot([Gaxt[t].T,\
                                       np.linalg.inv(sigmaa_tilde[t]),\
                                       Gat[t]])
            
            # Parametrization for F_t (cont.)
            Dt[t] = Gxxt[t] + np.dot(Gaxt[t].T,Eax) - \
                    0.5*np.linalg.multi_dot([A1t[t].T, tau, A1t[t]])
            Ht[t] = Gxt[t] - np.dot(Dax,bt) + \
                    np.linalg.multi_dot([A1t[t].T,gammab,Gat[t]]) + \
                    beta*np.linalg.multi_dot([Gaxt[t].T,\
                              np.linalg.inv(sigmaa_tilde[t]),Gat[t]])
            ft[t] = gt[t] - 0.5*np.linalg.multi_dot([bt.T,tau,bt]) -\
                    np.linalg.multi_dot([Gat[t].T,gammab,bt])+\
                    beta/2*np.linalg.multi_dot([Gat[t].T,\
                            np.linalg.inv(sigmaa_tilde[t]),Gat[t]])-Lb

            # Update of mean and variance
            sigmaa2 = np.linalg.inv(sigmaa_tilde[t])
            A0t[t] = abar[t] + np.linalg.multi_dot([sigmaa2, \
                                            np.linalg.inv(sigmaat[t]),\
                                            A0t[t]-abar[t]]) + \
                                            beta*np.dot(sigmaa2,\
                                              Gat[t]-\
                                              np.dot(Gaxt[t],xbar[t]))
            A1t[t] = np.dot(sigmaa2,(np.dot(np.linalg.inv(sigmaat[t]),\
                                            A1t[t])+beta*Gaxt[t]))
            sigmaat[t] = sigmaa2

        # Forward iteration to update trajectories
        for t in range(0,T-1):
            abar[t] = A0t[t]+np.dot(A1t[t], xbar[t])

        for t in range(0,T-1):
            xbar[t+1] = np.multiply(1+rt[t],xbar[t]+\
                                    np.dot(In.T,abar[t]))
        for t in range(0,T-1):
            abar[t] = 0*abar[t]
        # Set the terminal condition again
        duT = -xbar[T-2]-np.dot(In.T,abar[T-1])
        dat[T-1]=np.concatenate([np.maximum(duT,np.array([[0]]*N)),\
                                 np.maximum(-duT,np.array([[0]]*N))])

        #print (A0t[t],A1t[t])
        
    return (A0t,A1t)

###################### Model training ######################

# Set some parameters
train_window = 63
T_all=prices.shape[0]-train_window-1
N=prices.shape[1]-1 #exclude market portfolio
rf = (1+0.02)**(1/365)-1 #Risk-free rate
gamma = 0.01
beta = 0.01
iterations = 100

# Tune parameters through training dataset
# First 70% data as traning set
training_data_index = int(len(returns_all.index)*0.7)
data_train = returns_all.iloc[:training_data_index,]
data_test = returns_all.iloc[training_data_index:,]
A0_list_train = []
A1_list_train = []
for i in range(int(len(data_train.index)/train_window)-1):
    start = i * train_window
    end = start + train_window
    # Run the model
    A0, A1 = G_learning_with_linear_dynamics(start, end, data_train,\
                                             rf, gamma, beta,\
                                             iterations)
    A0_list_train.append(A0)
    A1_list_train.append(A1)

# Train the model on testing dataset
A0_list_test = []
A1_list_test = []
for i in range(int(len(data_test.index)/train_window)-1):
    start = i * train_window
    end = start + train_window
    # Run the model
    A0, A1 = G_learning_with_linear_dynamics(start, end, data_test, \
                                             rf, gamma, beta, \
                                             iterations)
    A0_list_test.append(A0)
    A1_list_test.append(A1)

###################### Backtesting ######################

# Backtesting: Clean the result and calculate the returns
returns_result = data_test.copy().iloc[train_window:,range(N)]

# Create columns
for i in range(N):
    returns_result['money in '+str(i)] = 0.5
returns_result['risk-free asset value'] = 1
returns_result['Portfolio Value'] = 0

# Some variables for convenience
I = np.identity(N)
In = np.concatenate((I,-I))
x = []
x_before_return = []
for t in range(len(returns_result.index)):
    x.append(np.array([[1/N]]*N))
    x_before_return.append(np.array([[1/N]]*N))

for t in range(len(returns_result.index)):
    j = int(t/train_window)
    k = t - j * train_window
    
    # Write the money invested to the testing dataset
    if (k>0):
        x_before_return[t]=x[t-1]+np.dot(In.T, A0_list_test[j][k-1]+\
                                         np.dot(A1_list_test[j][k-1],\
                                                x[t-1]))
        for i in range(N): 
            x[t][i] = (1+returns_result.iloc[t,i]) * \
                      x_before_return[t][i]
            returns_result.loc[returns_result.index[t],\
                               'money in '+str(i)] = x[t][i]
    # Calculate portfolio values and risk-free asset value
    if (t==0):
        returns_result.loc[returns_result.index[t],\
                           'Portfolio Value'] = 1
        returns_result.loc[returns_result.index[t],\
                           'risk-free asset value'] = 0
    else:
        returns_result.loc[returns_result.index[t],'risk-free asset value'] = \
            (returns_result.loc[returns_result.index[t-1],'Portfolio Value']-x_before_return[t].sum())*(1+rf)
        returns_result.loc[returns_result.index[t],'Portfolio Value'] = \
            x[t].sum() + returns_result.loc[returns_result.index[t],'risk-free asset value']

# Calculate daily return
returns_result['Daily return'] =  returns_result['Portfolio Value']/\
                            returns_result['Portfolio Value'].shift(1)
# Calculate allocation weights
for i in range(N):
    returns_result['w'+str(i)] = returns_result['money in '+str(i)]/\
                            returns_result['Portfolio Value']


returns_result.head()

returns_result.tail()

returns_result.to_csv('Continuous G-learning.csv')

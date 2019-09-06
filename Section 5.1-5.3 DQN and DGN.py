#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: heshan
"""

##################### Section 5.1-5.3 #####################

# Import libraries
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import time
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
import scipy.optimize as sco
from enums import *
import tensorflow as tf
import random

# Set path
path = '/Users/heshan/Desktop/Portfolio Optimization/'
os.chdir(path)

################### # Section 5.1 Data Visualization ###################
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

# Plot the closed prices for two stocks
fig = plt.figure(figsize=(14, 7))
plt.plot(prices.index, prices['AAPL'], lw=3, alpha=0.8)
plt.plot(prices.index, prices['GOOG'], lw=3, alpha=0.8)
plt.legend(labels = ('AAPL','GOOG'),loc='upper left')
plt.ylabel('Closed Prices')
plt.suptitle('Adjusted Closed Prices of Stocks')
fig.savefig("Orginal_Closed_Prices.pdf")

# Plot the returns for two stocks
fig = plt.figure(figsize=(14, 7))
plt.plot(prices.index, prices['AAPL']/prices['AAPL'][0],\
         lw=3, alpha=0.8)
plt.plot(prices.index, prices['GOOG']/prices['GOOG'][0],\
         lw=3, alpha=0.8)
plt.legend(labels = ('AAPL','GOOG'),loc='upper left')
plt.ylabel('Cumulative Returns')
plt.suptitle('Cumulative Returns of Stocks')
fig.savefig("Orginal_Cumulative_Return.pdf")

# Calculate returns from prices
returns = prices.pct_change()
returns.dropna(inplace=True)
returns.head()

# Set some parameters
T = returns.index.size
N = len(keylist) - 1
drop_window = 63 # Drop first 63 days for training

############# Benchmark 1: Equal Allocation Portfolio #################### Benchmark 1: 
# 50% weights in each stock at t=0, and do nothing afterwards.
Benchmark1 = returns.iloc[drop_window+1:,].copy()
w0 = 1/N

# Initialize columns
for i in range(N):
    Benchmark1[Benchmark1.columns[i]+' Cum return'] =0
Benchmark1['Daily return'] = 0
Benchmark1['Cum return'] = 0

# Get cumulative returns
for i in range(N):
    Benchmark1[Benchmark1.columns[i]+' Cum return'] = \
        (1+Benchmark1.iloc[:,i]).cumprod()

for i in range(N):
    Benchmark1['Cum return'] += \
        w0*Benchmark1[Benchmark1.columns[i]+' Cum return']

# Get daily returns
Benchmark1['Daily return'] = Benchmark1['Cum return'].pct_change()
Benchmark1['Daily return'][0] = Benchmark1['Cum return'][0] - 1
Benchmark1.tail(5)


############ Benchmark 2: Markowitz Portfolio ##############

# Reference:
# https://towardsdatascience.com/
# efficient-frontier-portfolio-optimisation-in-python-e7844051e7f

# Calculate portfolio return and standard deviation
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights )
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std, returns

# Calculate negative portfolio Sharpe ratio
def neg_sharpe_ratio(weights, mean_returns, cov_matrix,\
                     risk_free_rate):
    p_var, p_ret = portfolio_performance(weights,\
                                         mean_returns, cov_matrix)
    return (-(p_ret - risk_free_rate) / p_var)

# Optmize Sharpe ratio
def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    # Constraint 1: use all investment
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Constraint 2: long-only
    bound = (0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio,num_assets*[1./num_assets,],\
                          args=args, method='SLSQP', bounds=bounds,\
                          constraints=constraints)
    return result

# Calculate portfolio negative return
def neg_return(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    return (-p_ret)

# Optmize return
def max_return(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    # Constraint 1: use all investment
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Constraint 2: long-only
    bound = (0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_return, num_assets*[1./num_assets,],\
                          args=args,method='SLSQP', bounds=bounds,\
                          constraints=constraints)
    return result

alpha = 0.89 #Decay parameter fro EWMA

# Markowitz Portolio Optimization
markowitz = returns.copy()
rf = (1+0.01)**(1/365)-1
train_window = drop_window
for i in range(N):
    markowitz[markowitz.columns[i]+' Weight'] = 0

T = len(markowitz.index)
markowitz.head(5)

# markowitz_SR = markowitz.copy()

# Count eclipsing time
start_time = time.process_time()

# Rolling train and test data
# Use 63-day data to allocate weight for next day
# Use 63-day EWMA return to estimate future one-day 
# expectation and std of return, smoothing alpha=0.89

for t in range(T-train_window-1):
    train_index = markowitz.index[t:t+train_window]
    train_data = markowitz.loc[train_index,keylist[0:N]]
    # EWMA mean and cov to predict next day's mean return and cov
    mean_returns =train_data.ewm(alpha=alpha, \
                            min_periods=train_window).mean().iloc[-1,]
    cov_matrix = train_data.ewm(alpha=alpha, \
                            min_periods=train_window).cov().iloc[-N:,]

    for i in range(N):
        # Maximize Return
        markowitz.loc[markowitz.index[t+train_window+1],\
                      markowitz.columns[i]+' Weight'] = \
                      max_return(mean_returns, cov_matrix, rf).x[i]
        
        # Maximize Sharpe ratio
        # markowitz_SR.loc[markowitz.index[t+train_window+1],\
                      #markowitz.columns[i]+' Weight'] = \
                      #max_sharpe_ratio(mean_returns, \
                      #cov_matrix, rf).x[i]   
                      
end_time = time.process_time()
print (end_time-start_time)

markowitz = markowitz.iloc[drop_window+1:,].copy()
markowitz['Daily return'] = 0

for i in range(N):
    markowitz['Daily return'] += markowitz.iloc[:,i] * \
                    markowitz.loc[:,markowitz.columns[i]+' Weight']
markowitz['Cum return'] = (1+markowitz['Daily return']).cumprod()
markowitz.tail(5)

markowitz.to_csv('markowitz.csv')

# Plot cumulative returns for benchmarks
fig = plt.figure(figsize=(14, 7))
plt.plot(markowitz['Cum return'].index,\
         (1+markowitz['AAPL']).cumprod(), lw=2, alpha=0.8)
plt.plot(markowitz['Cum return'].index,\
         (1+markowitz['GOOG']).cumprod(), lw=2, alpha=0.8)
plt.plot(markowitz['Cum return'].index, markowitz['Cum return'],\
         lw=3, alpha=1,label='linear', linestyle = '--',\
         color = 'forestgreen')
plt.plot(Benchmark1['Cum return'].index, Benchmark1['Cum return'],\
         lw=3, alpha=1,label='linear', linestyle = '--',\
         color = 'darkred')
plt.legend(labels = ('AAPL','GOOG',\
                     'Markowitz with Objective of Maximizing Return',\
                     'Equal Allocation'),loc='upper left',fontsize=18)
plt.ylabel('Cumulative return')
plt.suptitle('Cumulative Returns of Benchmarks')
fig.savefig("Benchmarks.pdf")

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
    
rf = 0.01 # Assumption on annual risk-free rate
# Evaluate performance of two stocks
evaluate_performance(markowitz['AAPL'], rf)
evaluate_performance(markowitz['GOOG'], rf)

######################### Section 5.2: DQN ###################################

### Implementation of the model
# Clean data for DQN and DGN
# Parameter prices_state indicates whether include prices (True) 
# or returns (False) to the state space
def data_cleansing(data, start, end, historical_prices_no, \
                   book_value, prices_state = True):

    DQN_returns = data.copy()[start:end]
    DQN_N = 2
    
    # Past N day return or prices in state space
    for label in DQN_returns.columns[:DQN_N]:
        DQN_returns[label+' t'] = DQN_returns[label]
        for i in range(1,historical_prices_no+1):
            DQN_returns[label+' t-'+str(i)] = \
                                DQN_returns[label].shift(i)
    
    # Further clean data if return in state space
    if (prices_state == False):
        for label in DQN_returns.columns[:DQN_N]:
            DQN_returns[label+' t'] = DQN_returns[label+' t']\
                                    /DQN_returns[label+' t'].shift(1)-1
            for i in range(1,historical_prices_no+1):
                DQN_returns[label+' t-'+str(i)] = \
                        DQN_returns[label+' t-'+str(i)]/\
                        DQN_returns[label+' t-'+str(i)].shift(1)-1
    
    # Initialize weights and portfolio values
    DQN_returns['w0'] = DQN_returns['w1'] = 0.5
    DQN_returns['Portfolio Value'] = book_value
    
    # Stock daily returns
    for label in DQN_returns.columns[:DQN_N]:
        DQN_returns[label+' return'] = DQN_returns[label].pct_change()
        
    DQN_returns.dropna(inplace=True)
    DQN_returns.drop(columns='Market', inplace=True)
    
    # Positions (shares owned) 
    for i in range(DQN_N):
        DQN_returns[DQN_returns.columns[i]+' shares'] = \
            DQN_returns['Portfolio Value']*DQN_returns['w'+str(i)]/\
            (DQN_returns[DQN_returns.columns[i]]/\
            (1+DQN_returns[DQN_returns.columns[i]+' return']))
    
    return (DQN_returns)

# Reference: Modified from
# https://blog.valohai.com/reinforcement-learning-
# tutorial-basic-deep-q-learning
class DQN:
    # Price: discount 0.5, batch32
    # Return: discount 0.6 batch 32
    def __init__(self, DQN_returns, action_space, update_freq = 100,\
                 learning_rate=1e-3, discount=0.5, \
                 exploration_rate=0.95, iterations=100):
        
        self.learning_rate = learning_rate
        self.discount = discount # Discount factor
        # Initial exploration rate
        self.exploration_rate = exploration_rate 
        
        # Input is a state 
        self.input_count = (historical_prices_no+1)*2+3
        # Output is an action
        self.output_count = len(action_space)
        
        self.returns = DQN_returns # Original data for DQN
        self.action_space = action_space # Action space
        self.T = len(DQN_returns.index)
        self.iterations = iterations
        self.experience = []
        self.update_freq = update_freq # Frequency to update target Q
        self.reply_memory = 1000 # Experience memory queue sie
        self.batch = 32 # Batch size
        
        self.batch_count = 0 
        self.training_input_array = \
                    np.empty((self.batch,self.input_count))
        self.target_output_array = \
                    np.empty((self.batch,self.output_count))
        self.epsilon_min = 0.05 # Min epsilon
        self.epsilon_start = 0.95 # Max epsilon
        self.lr_control = 0 # Parameter to control the learning rate
        self.trackloss = np.array([[]]) # List to store losses
        self.drop_prob = 0.2 # Dropout rate
        
        self.session = tf.Session()
        self.define_model() 
        self.session.run(self.initializer)
        
    
    def define_model(self):
        # Define the neural network graph for primary Q
        with tf.variable_scope("Q_primary"):
            self.model_input_primary = \
                tf.placeholder(dtype=tf.float32, \
                               shape=[None,self.input_count])

            # Two hidden layers of 16 neurons with 
            # leaky_relu activation initialized by Gaussian
            fc1_primary = tf.layers.dense(self.model_input_primary, \
                              16, activation = tf.nn.leaky_relu, \
                              kernel_initializer = \
                                  tf.random_normal_initializer(\
                                  mean=0,stddev=1e-2))
            fc1_primary_after_dropout = tf.nn.dropout(fc1_primary,\
                              rate=self.drop_prob)
            fc2_primary = tf.layers.dense(fc1_primary_after_dropout,\
                              16, activation=tf.nn.leaky_relu, \
                              kernel_initializer=\
                                  tf.random_normal_initializer(\
                                  mean=0,stddev=1e-2))

            self.model_output_primary = tf.layers.dense(\
                              fc2_primary, self.output_count)

        # Same settings for target Q
        with tf.variable_scope("Q_target"):
            self.model_input_target = tf.placeholder(dtype=tf.float32,\
                                         shape=[None,self.input_count])

            # 2 hidden layers of 16 neurons with leaky_relu activation
            fc1_target = tf.layers.dense(self.model_input_target, \
                             16, activation=tf.nn.leaky_relu, \
                             kernel_initializer=\
                             tf.random_normal_initializer(\
                             mean=0,stddev=1e-2))
            fc1_target_after_dropout = tf.nn.dropout(\
                             fc1_target,rate=self.drop_prob)
            fc2_target = tf.layers.dense(fc1_target_after_dropout, \
                             16, activation=tf.nn.leaky_relu, \
                             kernel_initializer=\
                             tf.random_normal_initializer(\
                             mean=0,stddev=1e-2))

            self.model_output_target = tf.layers.dense(fc2_target, \
                                                   self.output_count)    
        
        # Network output
        self.target_output_primary = tf.placeholder(shape=\
                        [None, self.output_count], dtype=tf.float32)
        
        # Loss is MSE between primary and target Q
        self.loss = tf.losses.mean_squared_error(\
                 self.target_output_primary, self.model_output_primary)

        # Gradient descent optimizer to minimize loss
        self.optimizer_fast = tf.train.GradientDescentOptimizer(\
            learning_rate=self.learning_rate).minimize(self.loss)
        self.optimizer_mid = tf.train.GradientDescentOptimizer(\
            learning_rate=self.learning_rate/10).minimize(self.loss)
        self.optimizer_slow = tf.train.GradientDescentOptimizer(\
            learning_rate=self.learning_rate/50).minimize(self.loss)
        
        # Get all the variables in the Q primary network.
        self.q_primary_varlist_ = tf.get_collection(\
            tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_primary")
        # Get all the variables in the Q target network.
        self.q_target_varlist_ = tf.get_collection(\
            tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_target")
        assert len(self.q_primary_varlist_) == len(\
            self.q_target_varlist_)
        
        # Initialize weights 
        self.initializer = tf.global_variables_initializer()
        
        rate = 0.1
        self.q_target_update_ = tf.group(\
            [v_t.assign(v_t * (1 - rate) + v * rate)\
            for v_t, v in zip(self.q_target_varlist_, \
                              self.q_primary_varlist_)])
        
    # Get Q-value from neural network given state
    def get_Q(self, state, target = False):
        if target == True:
            return self.session.run(self.model_output_target, \
                    feed_dict={self.model_input_target: state})[0]
        else:
            return self.session.run(self.model_output_primary, \
                    feed_dict={self.model_input_primary: state})[0]
    
    # Get next action using epsilon greedy policy
    def get_next_action(self, state):
        # Exploration
        if random.random() > self.exploration_rate: 
            return self.greedy_action(state)
        else:
            return self.random_action()

    # Greedy policy
    def greedy_action(self, state):
        return np.argmax(self.get_Q(state))

    # Random exploration
    def random_action(self):
        return (random.randint(0,self.output_count-1))
    
    # Get reward given state and action
    def get_reward(self, action, t):
        w0 = self.action_space[action] 
        w1 = 1 - w0
        self.returns.iloc[t,4+(1+historical_prices_no)*2] = (w0 * \
             self.returns.iloc[t,5+(1+historical_prices_no)*2] + \
             w1*self.returns.iloc[t,6+(1+historical_prices_no)*2]+1)*\
             self.returns.iloc[t-1,4+(1+historical_prices_no)*2]
        reward = self.returns.iloc[t,4+(1+historical_prices_no)*2] - 1\
             - abs(self.action_space[action])*0.01*2*0
        
        return (reward)

    # Train the DQN with experience reply
    def train_experience_reply(self):
        '''
        Train the DQN, with
        1. experience reply
        2. two networks for primary and target Q respectively
        '''
        
        for t in range(0,self.T-1):
                        
            self.batch_count +=1

            # Get old and new states
            old_state = np.array([self.returns.iloc[t,list(range(2,2+\
                 (1+historical_prices_no)*2))+[4+(1+\
                 historical_prices_no)*2]+[7+(1+historical_prices_no)\
                 *2]+[8+(1+historical_prices_no)*2]]])
            old_state = old_state.reshape((1,self.input_count))
            
            # Get action
            action = self.get_next_action(old_state)
            
            # Move to next state
            self.returns.iloc[t+1,2+(1+historical_prices_no)*2] = \
                self.action_space[action] 
            self.returns.iloc[t+1,3+(1+historical_prices_no)*2] = \
                1-self.returns.iloc[t+1,2+(1+historical_prices_no)*2]
            self.returns.iloc[t+1,4+(1+historical_prices_no)*2] = \
                (self.returns.iloc[t+1,5+(1+historical_prices_no)*2]*\
                self.returns.iloc[t+1,2+(1+historical_prices_no)*2]+\
                self.returns.iloc[t+1,6+(1+historical_prices_no)*2]*\
                self.returns.iloc[t+1,3+(1+historical_prices_no)*2]+1)\
                *self.returns.iloc[t,4+(1+historical_prices_no)*2]
            self.returns.iloc[t+1,7+(1+historical_prices_no)*2] = \
                self.returns.iloc[t,4+(1+historical_prices_no)*2]*\
                self.returns.iloc[t+1,2+(1+historical_prices_no)*2]/\
                self.returns.iloc[t,0]
            self.returns.iloc[t+1,8+(1+historical_prices_no)*2] = \
                self.returns.iloc[t,4+(1+historical_prices_no)*2]*\
                self.returns.iloc[t+1,3+(1+historical_prices_no)*2]/\
                self.returns.iloc[t,1]
          
            new_state = np.array([self.returns.iloc[t+1,list(range(2,\
                2+(1+historical_prices_no)*2))+[4+(1+\
                historical_prices_no)*2]+[7+(1+historical_prices_no)\
                *2]+[8+(1+historical_prices_no)*2]]])
            new_state = new_state.reshape((1,self.input_count))

            # Get reward
            reward = self.get_reward(action, t+1)
            
            # Store experience (old_state,action,reward,new_state)
            self.experience.append([old_state,action,reward,new_state])
            if len(self.experience) > self.reply_memory:
                self.experience.pop(0)
            
            # Get an experience from memory randomly
            single_experience = random.choice(self.experience)
            old_state = single_experience[0]
            new_state = single_experience[3]
            action = single_experience[1]
            reward = single_experience[2]
             
            # Get old Q-value from primary Q
            old_state_Q_values = self.get_Q(old_state)
            # Get new Q-value from target Q
            new_state_Q_values = self.get_Q(new_state, True)
            
            # Update Q
            old_state_Q_values[action] = reward + self.discount *\
                np.amax(new_state_Q_values)
            
            # Setup training data
            self.training_input_array[self.batch_count-1] = \
                np.array([old_state])
            self.target_output_array[self.batch_count-1] = \
                np.array([old_state_Q_values])
            
            # Train the primary network every batch
            if self.batch_count == self.batch:
                    
                training_data = {self.model_input_primary: \
                                 self.training_input_array, \
                                 self.target_output_primary: \
                                 self.target_output_array}
                self.batch_count = 0

                # Decrease the learning rate gradually
                if (self.lr_control <= self.iterations/3):
                    self.session.run(self.optimizer_fast, \
                                     feed_dict=training_data)
                elif (self.lr_control <= 2*self.iterations/3):
                    self.session.run(self.optimizer_mid, \
                                     feed_dict=training_data)
                else:
                    self.session.run(self.optimizer_slow, \
                                     feed_dict=training_data)

                loss, _ = self.session.run([self.loss, \
                                        self.q_primary_varlist_[0]], \
                                        feed_dict = training_data)
                outout = self.session.run(self.model_output_primary, \
                                        feed_dict = training_data)
                #print (loss)
                # Store losses
                self.trackloss = np.append(self.trackloss, loss)
                
            # Update target network every update_freq steps
            if (t % self.update_freq == 0):
                self.session.run(self.q_target_update_)
    
    # Train the DQN for many iterations        
    def update_experience_reply_shift_exploration(self):
        for i in range (self.iterations):
            
            self.lr_control += 1
            #print('iteration: %d' % i)
            #print('epsilon: %f' % self.exploration_rate)
            self.train_experience_reply() # Train DQN

            # Gradually decrease the epsilon (exporation rate)
            if self.exploration_rate > 0:
                epsilon_decay = self.iterations / self.batch / 4
                self.exploration_rate = self.epsilon_min + \
                    (self.epsilon_start - self.epsilon_min) * \
                    np.exp(-1 * i/self.batch / epsilon_decay)
                   
# Find optimal actions from the DQN (Qlearning = True) or DGN (False)
def predict_actions(test_data, Deep_Q_Network, historical_prices_no, \
                    Qlearning = True):
    
    for t in range(len(test_data.index)-1):
        # Get state
        state = np.array([test_data.iloc[t,list(range(2,\
                          2+(1+historical_prices_no)*2))+\
                         [4+(1+historical_prices_no)*2]+\
                         [7+(1+historical_prices_no)*2]+\
                         [8+(1+historical_prices_no)*2]]])
        state = state.reshape((1,Deep_Q_Network.input_count))
        
        # For DQN
        if (Qlearning == True):
            action = np.argmax(Deep_Q_Network.get_Q(state))
        # For DGN
        else:
            action = Deep_Q_Network.get_G_action(state)
        
        # Write next state given state and action
        test_data.iloc[t+1,2+(1+historical_prices_no)*2] = \
            Deep_Q_Network.action_space[action] 
        test_data.iloc[t+1,3+(1+historical_prices_no)*2] = \
            1-test_data.iloc[t+1,2+(1+historical_prices_no)*2]
        test_data.iloc[t+1,4+(1+historical_prices_no)*2] = \
            (test_data.iloc[t+1,5+(1+historical_prices_no)*2]*\
             test_data.iloc[t+1,2+(1+historical_prices_no)*2]+\
             test_data.iloc[t+1,6+(1+historical_prices_no)*2]*\
             test_data.iloc[t+1,3+(1+historical_prices_no)*2]+1)*\
            test_data.iloc[t,4+(1+historical_prices_no)*2]
        test_data.iloc[t+1,7+(1+historical_prices_no)*2] = \
            test_data.iloc[t,4+(1+historical_prices_no)*2]*\
            test_data.iloc[t+1,2+(1+historical_prices_no)*2]/\
            test_data.iloc[t,0]
        test_data.iloc[t+1,8+(1+historical_prices_no)*2] = \
            test_data.iloc[t,4+(1+historical_prices_no)*2]*\
            test_data.iloc[t+1,3+(1+historical_prices_no)*2]/\
            test_data.iloc[t,1]
  
    return (test_data)

# Calculate the portfolio return from DQN or DGN
def get_portfolio_return(data):
    
    data['Portfolio Daily Return'] = data['AAPL return']*data['w0']+\
                                     data['GOOG return']*data['w1']
    data.loc[data.index[0],'Portfolio Daily Return'] = 0
    data['Portfolio Value'] = \
                        (1+data['Portfolio Daily Return']).cumprod()
    return (data)

### Training dataset to tune parameters
    # Train the DQN portfolio
# Count eclipse time
start_time = time.perf_counter()

# Set first 70% data as training data
training_data_index = int(len(prices.index)*0.7)

# Set some parameters
training = 63 # Rolling training window
use_price = True # Use prices or returns in the state space 
# Set training data
data = prices.copy().iloc[:training_data_index,]
historical_prices_no = 2 # Past 2-day prices or returns
book_value = 1 # Initial portfolio value
action_space = list(np.arange(0,1.05,0.1)) 
repeat_times = 1

# Train the DQN and output results as csv files
for iterations in range(repeat_times):
    
    # Inout and clean data
    DQN_returns_pred = data_cleansing(data,0,1,historical_prices_no,\
                                      book_value)
    
    for i in range(int(len(data.index)/training)-1):
        #-1 above is to leave space for testing data
        # Set start and end date for each rolling training dataset
        start = i * training
        end = start + training
        if (use_price==False):
            end += 1
        DQN_returns_train = data_cleansing(data, start, end, \
                                           historical_prices_no, \
                                           book_value, use_price)
        # Train the DQN 
        tf.reset_default_graph()
        Deep_Q = DQN(DQN_returns_train, action_space, iterations=100)
        # Price, Return iterations 100
        Deep_Q.update_experience_reply_shift_exploration()
        
        # Get optimal actions from the DQN
        if (use_price==False):
            DQN_returns_test = data_cleansing(data, end-1,\
                                              end + training,\
                                              historical_prices_no, \
                                              book_value, use_price)
        else:
            DQN_returns_test = data_cleansing(data, end, end+training,\
                                              historical_prices_no, \
                                              book_value, use_price)
        DQN_returns_test = predict_actions(DQN_returns_test, Deep_Q,\
                                           historical_prices_no)
        # Append data to a list
        DQN_returns_pred = DQN_returns_pred.append(DQN_returns_test)
    # Calcuate cumulative returns
    DQN_returns_pred = get_portfolio_return(DQN_returns_pred)
    # Output as csv files
    DQN_returns_pred.to_csv('DQN'+str(iterations)+'.csv')

end_time = time.perf_counter()
print (end_time - start_time)

# Plot the losses
fig = plt.figure(figsize=(14, 7))
plt.plot(Deep_Q.trackloss, lw=3, alpha=0.8)
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.suptitle('Loss of DQN')
fig.savefig("DQN_loss.pdf")

# Load results
repeat_times = 20
os.chdir(path+'Data Store/Q Train')

# Load results to calculate mean and variance
DQN_returns_pred= pd.DataFrame(index=DQN_returns_pred.index)
DQN_weights_pred= pd.DataFrame(index=DQN_returns_pred.index)

for iterations in range(repeat_times):
    i = pd.read_csv('DQN'+str(iterations)+'.csv', index_col='Date')
    DQN_returns_pred['Portfolio Value '+str(iterations)] = \
        i['Portfolio Value']
    DQN_weights_pred['w0 '+str(iterations)] = i['w0']

os.chdir(path)

# Mean and std for cumulative returns
DQN_returns_pred['Mean'] = DQN_returns_pred.mean(axis = 1)
DQN_returns_pred['Std'] = DQN_returns_pred.std(axis = 1)

# Mean and std for allocation weights
DQN_weights_pred['Mean'] = DQN_weights_pred.mean(axis = 1)
DQN_weights_pred['Std'] = DQN_weights_pred.std(axis = 1)

DQN_returns_pred['AAPL Daily Return'] = data.pct_change()['AAPL']
DQN_returns_pred['GOOG Daily Return'] = data.pct_change()['GOOG']
DQN_returns_pred.loc[DQN_returns_pred.index[0],'AAPL Daily Return'] = 0
DQN_returns_pred.loc[DQN_returns_pred.index[0],'GOOG Daily Return'] = 0
DQN_returns_pred['Benchmark'] = \
    (1+0.5*(DQN_returns_pred['AAPL Daily Return']+\
            DQN_returns_pred['GOOG Daily Return'])).cumprod()
DQN_returns_pred.tail()

DQN_weights_pred.tail()

# Plot cumulative returns of DQN vs benchmarks
fig = plt.figure(figsize=(14, 7))
plt.plot(DQN_returns_pred['Mean'], lw=3, alpha=0.8, color = 'blue')
plt.plot(DQN_returns_pred['Mean']+DQN_returns_pred['Std'],\
         linestyle = ':', lw=2, alpha=0.8, color = 'blue')
plt.plot(DQN_returns_pred['Mean']-DQN_returns_pred['Std'],\
         linestyle = ':', lw=2, alpha=0.8, color = 'blue')
plt.fill_between(DQN_returns_pred.index,\
                 DQN_returns_pred['Mean']-DQN_returns_pred['Std'],\
                 DQN_returns_pred['Mean']+DQN_returns_pred['Std'],\
                 color = 'lightsteelblue')
plt.plot(markowitz.loc[DQN_returns_pred.index,'Cum return'],lw=2, \
         alpha=0.8,label='linear',linestyle = '--', \
         color = 'forestgreen')
plt.plot(DQN_returns_pred.loc[DQN_returns_pred.index,'Benchmark'],\
         linestyle = '--', lw=2, alpha=0.8, color = 'darkred')
plt.legend(labels = ('DQN Mean','DQN Mean + Std','DQN Mean - Std',\
                     'Markowitz (Max Return)','Equal Allocation'),\
           loc='upper left')
plt.ylabel('Cumulative return')
plt.suptitle('Cumulative Returns of DQN (Training Data)')
fig.savefig("DQN_Benchmarks.pdf")



### Testing dataset to test results

# Count eclipse time
start_time = time.perf_counter()

# Set some parameters
training = 63 # Rolling training window
use_price = True # Use prices or returns in the state space 
# Set testing data
if (use_price==False):
    data = prices.copy().iloc[training_data_index-1:,]
else:
    data = prices.copy().iloc[training_data_index:,]
historical_prices_no = 2 # Past 2-day prices or returns
book_value = 1 # Initial portfolio value
action_space = list(np.arange(0,1.05,0.1))
repeat_times = 1

# Train the DQN and output results as csv files
for iterations in range(repeat_times):
    
    DQN_returns_pred = data_cleansing(data, 0, 1, \
                                      historical_prices_no, \
                                      book_value)

    for i in range(int(len(data.index)/training)-1): 
        #-1 above is to leave space for testing data
        # Set start and end date for each rolling training dataset
        start = i * training
        end = start + training
        if (use_price==False):
            end += 1
        
        DQN_returns_train = data_cleansing(data, start, end, \
                                           historical_prices_no, \
                                           book_value, use_price)      
        # Train the DQN model
        tf.reset_default_graph()
        #  Price, Return iterations 100
        Deep_Q = DQN(DQN_returns_train, action_space, iterations=100) 
        Deep_Q.update_experience_reply_shift_exploration()
        
        # Get optimal actions from DQN
        if (use_price==False):
            DQN_returns_test = data_cleansing(data,end-1,end+training,\
                                              historical_prices_no,\
                                              book_value, use_price)
        else:
            DQN_returns_test = data_cleansing(data, end, end+training,\
                                              historical_prices_no,\
                                              book_value, use_price)
        DQN_returns_test = predict_actions(DQN_returns_test, Deep_Q,\
                                           historical_prices_no)
        # Append results to a list
        DQN_returns_pred = DQN_returns_pred.append(DQN_returns_test)
    
    # Calculate cumulative returns for testing periods
    DQN_returns_pred = get_portfolio_return(DQN_returns_pred)
    # Outout results as csv files
    DQN_returns_pred.to_csv('DQN_test'+str(iterations)+'.csv')
    

end_time = time.perf_counter()
print (end_time - start_time)

# Load results
repeat_times = 20
#os.chdir(path+'Data Store/Q Test Return')
os.chdir(path+'Data Store/Q Test')

# Load results to calculate mean and variance
DQN_returns_pred_test = pd.DataFrame(index=DQN_returns_pred.index)
DQN_weights_pred_test = pd.DataFrame(index=DQN_returns_pred.index)

for iterations in range(repeat_times):
    i = pd.read_csv('DQN_test'+str(iterations)+'.csv', \
                    index_col='Date')
    DQN_returns_pred_test['Portfolio Value '+str(iterations)] = \
                    i['Portfolio Value']
    DQN_weights_pred_test['w0 '+str(iterations)] = i['w0']
os.chdir(path)

# Mean and std for cumulative returns
DQN_returns_pred_test['Mean'] = DQN_returns_pred_test.mean(axis = 1)
DQN_returns_pred_test['Std'] = DQN_returns_pred_test.std(axis = 1)
# Mean and std for allocation weights
DQN_weights_pred_test['Mean'] = DQN_weights_pred_test.mean(axis = 1)
DQN_weights_pred_test['Std'] = DQN_weights_pred_test.std(axis = 1)

DQN_returns_pred_test['AAPL Daily Return'] = data.pct_change()['AAPL']
DQN_returns_pred_test['GOOG Daily Return'] = data.pct_change()['GOOG']
DQN_returns_pred_test.loc[DQN_returns_pred_test.index[0],\
                          'AAPL Daily Return'] = 0
DQN_returns_pred_test.loc[DQN_returns_pred_test.index[0],\
                          'GOOG Daily Return'] = 0
DQN_returns_pred_test['Benchmark'] = \
    (1+0.5*(DQN_returns_pred_test['AAPL Daily Return']+\
            DQN_returns_pred_test['GOOG Daily Return'])).cumprod()

DQN_weights_pred_test.tail()

DQN_returns_pred_test.tail()

# Plot cumulative returns of DQN vs benchmarks
fig = plt.figure(figsize=(14, 7))
plt.plot(DQN_returns_pred_test['Mean'], lw=3, color = 'blue')
plt.plot(DQN_returns_pred_test['Mean']+DQN_returns_pred_test['Std'],\
         linestyle = ':', lw=2, alpha=0.8, color = 'blue')
plt.plot(DQN_returns_pred_test['Mean']-DQN_returns_pred_test['Std'],\
         linestyle = ':', lw=2, alpha=0.8, color = 'blue')
plt.fill_between(DQN_returns_pred_test.index, \
                 DQN_returns_pred_test['Mean']-\
                 DQN_returns_pred_test['Std'],\
                 DQN_returns_pred_test['Mean']+\
                 DQN_returns_pred_test['Std'], \
                 color = 'lightsteelblue')
plt.plot(markowitz.loc[DQN_returns_pred_test.index,'Cum return']/\
         markowitz.loc[DQN_returns_pred_test.index[0],'Cum return'],\
         lw=2, alpha=0.8,label='linear',linestyle = '--', \
         color = 'forestgreen')
plt.plot(DQN_returns_pred_test.loc[DQN_returns_pred.index,\
         'Benchmark'],linestyle = '--', lw=2, alpha=0.8, \
         color = 'darkred')
plt.legend(labels = ('DQN Mean','DQN Mean + Std','DQN Mean - Std',\
                     'Markowitz (Max Return)','Equal Allocation'),\
                    loc='upper left',borderaxespad=0., fontsize=15.5)
plt.ylabel('Cumulative return')
plt.suptitle('Cumulative Returns of DQN (Testing Data)')
fig.savefig("DQN_Benchmarks_Test.pdf")

# Get daily returns from cumlative returns
def to_daily_return (data):
    dailt_return=data/data.shift(1)-1
    dailt_return.dropna(inplace=True)
    return (dailt_return)

# Evaluate performances of portfolios 
rf = 0.01
evaluate_performance(to_daily_return(DQN_returns_pred_test['Mean']), \
                     rf)
evaluate_performance(to_daily_return(DQN_returns_pred_test['Mean']+\
                                     DQN_returns_pred_test['Std']), rf)
evaluate_performance(to_daily_return(DQN_returns_pred_test['Mean']-\
                                     DQN_returns_pred_test['Std']), rf)
evaluate_performance(to_daily_return(DQN_returns_pred_test['Benchmark']),\
                     rf)
evaluate_performance(to_daily_return\
                     (markowitz.loc[DQN_returns_pred_test.index,\
                     'Cum return']/markowitz.loc\
                      [DQN_returns_pred_test.index[0],'Cum return']),\
                     rf)

# Plot mean weigths of DQN portfolios
fig = plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(DQN_weights_pred_test['Mean'])
plt.ylabel('Weights of AAPL')
plt.xticks(rotation=30)
plt.subplot(1, 2, 2)
sns.distplot(DQN_weights_pred_test['Mean'], kde=False)
plt.ylabel('Frequency')
plt.xlabel('Weights of AAPL')
plt.suptitle('DQN Allocation Weights of AAPL (Testing Data)')
fig.savefig("DQN_Weights_Test.pdf")

###################### Section 5.3 DGN ############################

### Implmentation of the model

# Reference: Modified from
# https://blog.valohai.com/reinforcement-learning-
# tutorial-basic-deep-q-learning
class DGN:
    # Price discount 0.4, batch 32
    # Return discount 0.7 batch 16
    def __init__(self, prior, beta, beta_rate, DGN_returns, \
                 action_space, update_freq = 100, learning_rate=1e-3,\
                 discount=0.4, exploration_rate=0.95, iterations=100):
        
        self.learning_rate = learning_rate #Learning rate
        self.discount = discount  # Discount factor
        # Initial exploration rate
        self.exploration_rate = exploration_rate
        
        # Input is a state
        self.input_count = (historical_prices_no+1)*2+3
        # Output is an action
        self.output_count = len(action_space)
        
        self.returns = DGN_returns # Original data for DGN
        self.action_space = action_space # Action space
        self.T = len(DGN_returns.index)
        self.iterations = iterations
        self.experience = []
        self.update_freq = update_freq # Update frequency for target G
        self.reply_memory = 1000 # Size for experience memory queue
        self.batch = 32 # Batch size
        self.batch_count = 0

        # Bacth of input
        self.training_input_array = \
            np.empty((self.batch,self.input_count)) 
        # Bacth of output
        self.target_output_array = \
            np.empty((self.batch,self.output_count))
        self.prior = prior # Prior policy
        self.beta_rate = beta_rate # Increment of beta per iteration
        self.beta = beta # Initial beta
        
        self.epsilon_min = 0.05 # Min epsilon
        self.epsilon_start = 0.95 # Max epsilon
        self.lr_control = 0 # # Parameter to control the learning rate 
        self.drop_prob = 0.2 # Dropout rate         
        self.trackloss = np.array([[]]) # List to store losses
        
        self.session = tf.Session()
        self.define_model()
        self.session.run(self.initializer)
        
    def define_model(self):
        # Define the neural network graph for primary G
        with tf.variable_scope("G_primary"):

            self.model_input_primary=tf.placeholder(dtype=tf.float32, \
                                        shape=[None,self.input_count])

            # 2 hidden layers of 16 neurons with leaky_relu activation\
            # initialized by Gaussian
            fc1_primary = tf.layers.dense(self.model_input_primary,16,\
                                          activation=tf.nn.leaky_relu,\
                                          kernel_initializer=\
                                          tf.random_normal_initializer\
                                          (mean=0,stddev=1e-2))
            fc1_primary_after_dropout = tf.nn.dropout(fc1_primary,\
                                                  rate=self.drop_prob)
            fc2_primary=tf.layers.dense(fc1_primary_after_dropout,16,\
                                        activation=tf.nn.leaky_relu,\
                                        kernel_initializer=\
                                        tf.random_normal_initializer\
                                        (mean=0,stddev=1e-2))

            self.model_output_primary = tf.layers.dense(fc2_primary,\
                                                    self.output_count)

        # Same settings for target G
        with tf.variable_scope("G_target"):

            self.model_input_target = tf.placeholder(dtype=tf.float32,\
                                         shape=[None,self.input_count])

            # 2 hidden layers of 16 neurons with leaky_relu activation
            fc1_target = tf.layers.dense(self.model_input_target, 16,\
                                         activation=tf.nn.leaky_relu,\
                                         kernel_initializer=\
                                         tf.random_normal_initializer\
                                         (mean=0,stddev=1e-2))
            fc1_target_after_dropout = tf.nn.dropout(fc1_target,\
                                                 rate=self.drop_prob)
            fc2_target = tf.layers.dense(fc1_target_after_dropout, 16,\
                                         activation=tf.nn.leaky_relu,\
                                         kernel_initializer=\
                                         tf.random_normal_initializer\
                                         (mean=0,stddev=1e-2))

            self.model_output_target = tf.layers.dense(fc2_target, \
                                                   self.output_count)    
        
        # Network output
        self.target_output_primary = tf.placeholder(shape=[None,\
                 self.output_count], dtype=tf.float32)
        
        # Loss is MSE between primary and target G
        self.loss = tf.losses.mean_squared_error(\
                 self.target_output_primary, self.model_output_primary)
        # Decrease learning rate gradually
        self.optimizer_fast = tf.train.GradientDescentOptimizer\
            (learning_rate=self.learning_rate).minimize(self.loss)
        self.optimizer_mid = tf.train.GradientDescentOptimizer\
            (learning_rate=self.learning_rate/10).minimize(self.loss)
        self.optimizer_slow = tf.train.GradientDescentOptimizer\
            (learning_rate=self.learning_rate/50).minimize(self.loss)
        
        # Get all the variables in the G primary network.
        self.g_primary_varlist_ = tf.get_collection\
            (tf.GraphKeys.GLOBAL_VARIABLES, scope="G_primary")
        # Get all the variables in the G target network.
        self.g_target_varlist_ = tf.get_collection\
            (tf.GraphKeys.GLOBAL_VARIABLES, scope="G_target")
        assert len(self.g_primary_varlist_) == \
            len(self.g_target_varlist_)
        
        # Initializer to set weights to initial values
        self.initializer = tf.global_variables_initializer()
        
        # Update target G
        rate = 0.1
        self.g_target_update_ = tf.group(\
                    [v_t.assign(v_t * (1 - rate) + v * rate)\
                    for v_t, v in zip(self.g_target_varlist_, \
                                      self.g_primary_varlist_)])
       
    # Get G value from DGN given state
    # Target = True to get target G
    def get_G(self, state, target = False):
        if target == True:
            return self.session.run(self.model_output_target, \
                        feed_dict={self.model_input_target: state})[0]
        else:
            return self.session.run(self.model_output_primary, \
                        feed_dict={self.model_input_primary: state})[0]
    
    # Epsilon-greedy policy
    def get_next_action(self, state):
        if random.random() > self.exploration_rate: 
            return self.get_G_action(state)
        else:
            return self.random_action()

    # Get the distribution of actions given state
    def get_action_dist(self, state):
        state_G_values = self.get_G(state)
        pi_a_given_s = []
        for i in range(self.output_count):
            pi_a_given_s.append(self.prior[i]*\
                                np.exp(-self.beta*state_G_values[i]))
        pi_a_given_s = pi_a_given_s/np.sum(pi_a_given_s)
        return (pi_a_given_s)
    
    # Greedy policy for exploitation
    def get_G_action(self, state):
        pi_a_given_s = self.get_action_dist(state)
        random_number = random.random()
        for i in range(len(pi_a_given_s)):
            if (random_number <= pi_a_given_s[i]):
                return (i)
            else:
                random_number -= pi_a_given_s[i]
            
        return np.argmin(self.get_G(state))
    
    # Random policy for exploration
    def random_action(self):
        return (random.randint(0,self.output_count-1))
    
    # Get reward given state and action
    def get_reward(self, action, t):
        w0 = self.action_space[action] 
        w1 = 1 - w0
        self.returns.iloc[t,4+(1+historical_prices_no)*2] = \
            (w0*self.returns.iloc[t,5+(1+historical_prices_no)*2] + \
            w1*self.returns.iloc[t,6+(1+historical_prices_no)*2]+1)*\
            self.returns.iloc[t-1,4+(1+historical_prices_no)*2]
        reward = -(self.returns.iloc[t,4+(1+historical_prices_no)*2]-\
                   1 - abs(self.action_space[action])*0.01*2*0)      
        return (reward)
    
    # Train the DGN with experience reply
    def train_experience_reply(self):
        '''
        Train the DGN, with
        1. experience reply
        2. two networks for primary and target Q respectively
        '''
        
        for t in range(0,self.T-1):
                        
            self.batch_count +=1
            self.beta += self.beta_rate
            
            # Get old and new states
            old_state = np.array([self.returns.iloc[t,list(range(2,2+\
                 (1+historical_prices_no)*2))+\
                 [4+(1+historical_prices_no)*2]+\
                 [7+(1+historical_prices_no)*2]+\
                 [8+(1+historical_prices_no)*2]]])
            old_state = old_state.reshape((1,self.input_count))
            
            # Get action
            action = self.get_next_action(old_state)
            
            # Move to next state
            self.returns.iloc[t+1,2+(1+historical_prices_no)*2] = \
                self.action_space[action] 
            self.returns.iloc[t+1,3+(1+historical_prices_no)*2] = \
                1-self.returns.iloc[t+1,2+(1+historical_prices_no)*2]
            self.returns.iloc[t+1,4+(1+historical_prices_no)*2] = \
                (self.returns.iloc[t+1,5+(1+historical_prices_no)*2]*\
                self.returns.iloc[t+1,2+(1+historical_prices_no)*2]+\
                self.returns.iloc[t+1,6+(1+historical_prices_no)*2]*\
                self.returns.iloc[t+1,3+(1+historical_prices_no)*2]+1)\
                *self.returns.iloc[t,4+(1+historical_prices_no)*2]
            self.returns.iloc[t+1,7+(1+historical_prices_no)*2] = \
                self.returns.iloc[t,4+(1+historical_prices_no)*2]*\
                self.returns.iloc[t+1,2+(1+historical_prices_no)*2]/\
                self.returns.iloc[t,0]
            self.returns.iloc[t+1,8+(1+historical_prices_no)*2] = \
                self.returns.iloc[t,4+(1+historical_prices_no)*2]*\
                self.returns.iloc[t+1,3+(1+historical_prices_no)*2]/\
                self.returns.iloc[t,1]
             
            new_state = np.array([self.returns.iloc[t+1,list(range(2,\
                2+(1+historical_prices_no)*2))+\
                [4+(1+historical_prices_no)*2]+\
                [7+(1+historical_prices_no)*2]+\
                [8+(1+historical_prices_no)*2]]])
            new_state = new_state.reshape((1,self.input_count))

            # Get reward
            reward = self.get_reward(action, t+1)
            
            # Store experience (old_state,action,reward,new_state)
            self.experience.append([old_state,action,reward,new_state])
            if len(self.experience) > self.reply_memory:
                self.experience.pop(0)
            
            # Get an experience from memory randomly
            single_experience = random.choice(self.experience)
            old_state = single_experience[0]
            new_state = single_experience[3]
            action = single_experience[1]
            reward = single_experience[2]
            
            # Get old G from primary network
            old_state_G_values = self.get_G(old_state)
            # Get new G from target network
            new_state_G_values = self.get_G(new_state, True)
            
            # Update G            
            td = 0
            for i in range(self.output_count):
                td += self.prior[i]*np.exp(-self.beta*\
                                           new_state_G_values[i])
            old_state_G_values[action] = reward - self.discount / \
                                         self.beta * np.log(td)  
           
            # Setup training data
            self.training_input_array[self.batch_count-1] = \
                                         np.array([old_state])
            self.target_output_array[self.batch_count-1] = \
                                         np.array([old_state_G_values])
            
            # Train the primary network every batch
            if self.batch_count == self.batch:
                    
                training_data = {self.model_input_primary: \
                                 self.training_input_array, \
                                 self.target_output_primary: \
                                 self.target_output_array}
                self.batch_count = 0

                # Decrease the learning rate gradually
                if (self.lr_control <= self.iterations/3):
                    self.session.run(self.optimizer_fast, \
                                     feed_dict=training_data)
                elif (self.lr_control <= 2*self.iterations/3):
                    self.session.run(self.optimizer_mid, \
                                     feed_dict=training_data)
                else:
                    self.session.run(self.optimizer_slow, \
                                     feed_dict=training_data)
                 
                loss, a = self.session.run([self.loss, \
                                       self.g_primary_varlist_[0]],\
                                       feed_dict = training_data)
                outout = self.session.run(self.model_output_primary,\
                                          feed_dict = training_data)
                #print (loss)
                # Store the loss in the list
                self.trackloss = np.append(self.trackloss, loss)
            # Update target G every update_freq steps
            if (t % self.update_freq == 0):
                self.session.run(self.g_target_update_)
        
    # Train the DQN for many iterations  
    def update_experience_reply_shift_exploration(self):
        for i in range (self.iterations):
            
            self.lr_control += 1
            #print('iteration: %d' % i)
            #print('epsilon: %f' % self.exploration_rate)
            self.train_experience_reply()

            # Decrease the epsilon (exploration rate) geadually
            if self.exploration_rate > 0:
                epsilon_decay = self.iterations / self.batch / 4
                self.exploration_rate = self.epsilon_min + \
                    (self.epsilon_start - self.epsilon_min) * \
                    np.exp (-1 * i/self.batch / epsilon_decay)
                   

### Training dataset to tune parameters

# Set some parameters
training_data_index = int(len(prices.index)*0.7)
training = 63
data = prices.copy().iloc[:training_data_index,]
historical_prices_no = 2
book_value = 1
action_space = list(np.arange(0,1.05,0.1))
repeat_times = 1
use_price = True

# Train the DGN model
for iterations in range(repeat_times):
    
    DGN_returns_pred = data_cleansing(data, 0, 1, historical_prices_no, book_value)
    beta_rate = 1*1e-2
    beta = 0
    prior = [1/len(action_space)]*len(action_space)

    for i in range(int(len(data.index)/training)-1): 

        start = i * training
        end = start + training
        if (use_price==False):
            end += 1
        DGN_returns_train = data_cleansing(data, start, end, historical_prices_no, book_value, use_price)
        
        # Train the DGN model
        tf.reset_default_graph()
        Deep_G = DGN(prior, beta, beta_rate, DGN_returns_train, action_space, iterations=100)
        Deep_G.update_experience_reply_shift_exploration()
        
        # Find optimal actions by DGN
        if (use_price==False):
            DGN_returns_test = data_cleansing(data, end-1, end+training, historical_prices_no, book_value, use_price)
        else:
            DGN_returns_test = data_cleansing(data, end, end+training, historical_prices_no, book_value, use_price)
        DGN_returns_test = predict_actions(DGN_returns_test, Deep_G, historical_prices_no, False)
        DGN_returns_pred = DGN_returns_pred.append(DGN_returns_test)
    # Calculate portfolio returns
    DGN_returns_pred = get_portfolio_return(DGN_returns_pred)
    # Outout results to csv files
    DGN_returns_pred.to_csv('DGN'+str(iterations)+'.csv')

# Check the final beta
Deep_G.beta

# Plot losses of DGN
fig = plt.figure(figsize=(14, 7))
plt.plot(Deep_G.trackloss, lw=3, alpha=0.8)
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.suptitle('Loss of DGN')
fig.savefig("DGN_loss.pdf")

# Load results for 20 DGN runs
repeat_times = 20
os.chdir(path+'Data Store/G Train')

DGN_returns_pred = pd.DataFrame(index=DGN_returns_pred.index)

for iterations in range(repeat_times):
    i = pd.read_csv('DGN'+str(iterations)+'.csv', index_col='Date')
    DGN_returns_pred['Portfolio Value '+str(iterations)] = i['Portfolio Value']
os.chdir(path)

# Calculate mean and std of cumulative returns
DGN_returns_pred['Mean'] = DGN_returns_pred.mean(axis = 1)
DGN_returns_pred['Std'] = DGN_returns_pred.std(axis = 1)

# Calculate the equal allocation portfolio
DGN_returns_pred['AAPL Daily Return'] = data.pct_change()['AAPL']
DGN_returns_pred['GOOG Daily Return'] = data.pct_change()['GOOG']
DGN_returns_pred.loc[DGN_returns_pred.index[0],'AAPL Daily Return'] = 0
DGN_returns_pred.loc[DGN_returns_pred.index[0],'GOOG Daily Return'] = 0
DGN_returns_pred['Benchmark'] = (1+0.5*(DGN_returns_pred['AAPL Daily Return']+\
                                        DGN_returns_pred['GOOG Daily Return'])).cumprod()

DGN_returns_pred.tail()

# Plot figure of cumulative returns of DGN
fig = plt.figure(figsize=(14, 7))
plt.plot(DGN_returns_pred['Mean'], lw=3, color = 'blue')
plt.plot(DGN_returns_pred['Mean']+DGN_returns_pred['Std'],linestyle = ':', lw=2, alpha=0.8, color = 'blue')
plt.plot(DGN_returns_pred['Mean']-DGN_returns_pred['Std'],linestyle = ':', lw=2, alpha=0.8, color = 'blue')
plt.fill_between(DGN_returns_pred.index, DGN_returns_pred['Mean']-DGN_returns_pred['Std'],\
                 DGN_returns_pred['Mean']+DGN_returns_pred['Std'], color = 'lightsteelblue')
plt.plot(markowitz.loc[DGN_returns_pred.index,'Cum return'],\
         lw=2, alpha=0.8,label='linear',linestyle = '--', color = 'forestgreen')
plt.plot(DGN_returns_pred.loc[DGN_returns_pred.index,'Benchmark'],\
         linestyle = '--', lw=2, alpha=0.8, color = 'darkred')
plt.legend(labels = ('DGN Mean','DGN Mean + Std','DGN Mean - Std',\
                     'Markowitz with Objective of Maximizing Return','Equal Allocation'),loc='upper left')
plt.ylabel('Cumulative return')
plt.suptitle('Cumulative Returns of DGN (Training Data)')
fig.savefig("DGN_Benchmarks.pdf")



# Test dataset to test results

# Set some parameters
training = 63
use_price = True
if (use_price==False):
    data = prices.copy().iloc[training_data_index-1:,]
else:
    data = prices.copy().iloc[training_data_index:,]
historical_prices_no = 2
book_value = 1
action_space = list(np.arange(0,1.05,0.1))
repeat_times = 1

# Train the DGN on testing data
for iterations in range(repeat_times):
    
    DGN_returns_pred = data_cleansing(data, 0, 1, historical_prices_no, book_value)
    # Set beta and prior
    beta_rate = 1*1e-2
    beta = 0
    prior = [1/len(action_space)]*len(action_space)

    for i in range(int(len(data.index)/training)-1): 

        start = i * training
        end = start + training
        if (use_price==False):
            end += 1
        
        DGN_returns_train = data_cleansing(data, start, end, historical_prices_no, book_value, use_price)
        # Train the DGN model
        tf.reset_default_graph()
        Deep_G = DGN(prior, beta, beta_rate, DGN_returns_train, action_space, iterations=250) 
        # Iterations 250 for both prices and returns
        Deep_G.update_experience_reply_shift_exploration()

        # Get optimal actions from DGN
        if (use_price==False):
            DGN_returns_test = data_cleansing(data, end-1, end+training, historical_prices_no, book_value, use_price)
        else:
            DGN_returns_test = data_cleansing(data, end, end+training, historical_prices_no, book_value, use_price)          
        DGN_returns_test = predict_actions(DGN_returns_test, Deep_G, historical_prices_no, False)
        DGN_returns_pred = DGN_returns_pred.append(DGN_returns_test)
    
    # Calculate portfolio returns
    DGN_returns_pred = get_portfolio_return(DGN_returns_pred)
    # Output results to csv files
    DGN_returns_pred.to_csv('DGN_test'+str(iterations)+'.csv')

# Load results
repeat_times = 20
#os.chdir(path+'Data Store/G Test Return')
os.chdir(path+'Data Store/G Test')

DGN_returns_pred_test = pd.DataFrame(index=DGN_returns_pred.index)
DGN_weights_pred_test = pd.DataFrame(index=DGN_returns_pred.index)

for iterations in range(repeat_times):
    i = pd.read_csv('DGN_test'+str(iterations)+'.csv', index_col='Date')
    DGN_returns_pred_test['Portfolio Value '+str(iterations)] = i['Portfolio Value']
    DGN_weights_pred_test['w0 '+str(iterations)] = i['w0']
os.chdir(path)

# Calculate mean and std cumulative returns
DGN_returns_pred_test['Mean'] = DGN_returns_pred_test.mean(axis = 1)
DGN_returns_pred_test['Std'] = DGN_returns_pred_test.std(axis = 1)
# Calculate mean and std allocation weights
DGN_weights_pred_test['Mean'] = DGN_weights_pred_test.mean(axis = 1)
DGN_weights_pred_test['Std'] = DGN_weights_pred_test.std(axis = 1)

# Calculate the equal allocation portfolio
DGN_returns_pred_test['AAPL Daily Return'] = data.pct_change()['AAPL']
DGN_returns_pred_test['GOOG Daily Return'] = data.pct_change()['GOOG']
DGN_returns_pred_test.loc[DGN_returns_pred_test.index[0],'AAPL Daily Return'] = 0
DGN_returns_pred_test.loc[DGN_returns_pred_test.index[0],'GOOG Daily Return'] = 0
DGN_returns_pred_test['Benchmark'] = (1+0.5*(DGN_returns_pred_test['AAPL Daily Return']+\
                                             DGN_returns_pred_test['GOOG Daily Return'])).cumprod()

DGN_weights_pred_test.tail()

DQN_returns_pred_test.tail()

#Plot cumulative returns of DGN 
fig = plt.figure(figsize=(14, 7))
plt.plot(DGN_returns_pred_test['Mean'], lw=3, alpha=0.8, color = 'blue')
plt.plot(DGN_returns_pred_test['Mean']+DGN_returns_pred_test['Std'],linestyle = ':', lw=2, alpha=0.8, color = 'blue')
plt.plot(DGN_returns_pred_test['Mean']-DGN_returns_pred_test['Std'],linestyle = ':', lw=2, alpha=0.8, color = 'blue')
plt.fill_between(DGN_returns_pred_test.index, DGN_returns_pred_test['Mean']-DGN_returns_pred_test['Std'],\
                 DGN_returns_pred_test['Mean']+DGN_returns_pred_test['Std'], color = 'lightsteelblue')
plt.plot(markowitz.loc[DGN_returns_pred_test.index,'Cum return']/\
         markowitz.loc[DGN_returns_pred_test.index[0],'Cum return'],\
         lw=2, alpha=0.8,label='linear',linestyle = '--', color = 'forestgreen')
plt.plot(DGN_returns_pred_test.loc[DGN_returns_pred_test.index,'Benchmark'],\
         linestyle = '--', lw=2, alpha=0.8, color = 'darkred')
plt.legend(labels = ('DGN Mean','DGN Mean + Std','DGN Mean - Std',\
                     'Markowitz (Max Return)','Equal Allocation'),loc='upper left',\
                     borderaxespad=0., fontsize=15.5)
plt.ylabel('Cumulative return')
plt.suptitle('Cumulative Returns of DGN (Testing Data)')
fig.savefig("DGN_Benchmarks_Test.pdf")

# Evaluate portfolio performances of DGN and benchmarks
rf = 0.01
evaluate_performance(to_daily_return(DGN_returns_pred_test['Mean']), rf)
evaluate_performance(to_daily_return(DGN_returns_pred_test['Mean']+DGN_returns_pred_test['Std']), rf)
evaluate_performance(to_daily_return(DGN_returns_pred_test['Mean']-DGN_returns_pred_test['Std']), rf)
evaluate_performance(to_daily_return(DGN_returns_pred_test['Benchmark']), rf)
evaluate_performance(to_daily_return(markowitz.loc[DGN_returns_pred_test.index,'Cum return']/\
         markowitz.loc[DGN_returns_pred_test.index[0],'Cum return']), rf)

# Plot the mean weights of DGN portfolio
fig = plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(DGN_weights_pred_test['Mean'])
plt.ylabel('Weights of AAPL')
plt.xticks(rotation=30)
plt.subplot(1, 2, 2)
sns.distplot(DGN_weights_pred_test['Mean'], kde=False)
plt.ylabel('Frequency')
plt.xlabel('Weights of AAPL')
plt.suptitle('DGN Allocation Weights of AAPL (Testing Data)')
fig.savefig("DGN_Weights_Test.pdf")



### Check the optimal policy for very small beta

# Check the optimal policy for very small beta
training = 63
use_price = True
if (use_price==False):
    data = prices.copy().iloc[training_data_index-1:,]
else:
    data = prices.copy().iloc[training_data_index:,]
historical_prices_no = 2
book_value = 1
action_space = list(np.arange(0,1.05,0.1))
DGN_returns_pred = data_cleansing(data, 0, 1, historical_prices_no, book_value)
beta_rate = 0 # Zero update rate of beta
beta = 1e-5 # Initial beta
prior = [1/len(action_space)]*len(action_space)

for i in range(int(len(data.index)/training)-1):
    start = i * training
    if (use_price==False):
        start += 1
    end = start + training
    DGN_returns_train = data_cleansing(data, start, end, historical_prices_no, book_value, use_price)
    
    # Train the DGN with small beta
    tf.reset_default_graph()
    Deep_G = DGN(prior, beta, beta_rate, DGN_returns_train, action_space, iterations=250)
    Deep_G.update_experience_reply_shift_exploration()
    
    if (use_price==False):
        DGN_returns_test = data_cleansing(data, end-1, end+training, historical_prices_no, book_value, use_price)
    else:
        DGN_returns_test = data_cleansing(data, end-1, end+training, historical_prices_no, book_value, use_price)
    DGN_returns_test = predict_actions(DGN_returns_test, Deep_G, historical_prices_no, False)
    DGN_returns_pred = DGN_returns_pred.append(DGN_returns_test)

# Calculate cumlative returns and output a csv file
DGN_returns_pred = get_portfolio_return(DGN_returns_pred)
DGN_returns_pred.to_csv('DGN_test_small_beta.csv')

# Calculate the distribution of the optimal policy given a state
t = 6
state = np.array([Deep_G.returns.iloc[t,list(range(2,2+(1+historical_prices_no)*2))+\
                                      [4+(1+historical_prices_no)*2]+[7+(1+historical_prices_no)*2]+\
                                      [8+(1+historical_prices_no)*2]]])
dist = Deep_G.get_action_dist(state)
print('Final Beta: %f' % Deep_G.beta)
print(dist) # Almost deterministic

# Plot the distribution of optimal policy
fig = plt.figure(figsize=(14, 7))
plt.stem(np.array(Deep_G.action_space), dist, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.ylabel('p.m.f.')
plt.xlabel('Weights of AAPL')
plt.suptitle('DGN Allocation Weights of AAPL with Very Small Beta')
fig.savefig("DGN_Weights_Small_Beta.pdf")



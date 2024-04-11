#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats
from scipy import stats
from scipy.stats import t, norm
import math
import statistics
import warnings
warnings.filterwarnings("ignore")


# In[2]:


pwd


# # Normality Part (Wanting)

# In[3]:


# get, clean the dataset and check for any anomalies
tickers_list = ['^GSPC']
norm_data = yf.download(tickers_list,start = '1957-03-04', end = '2022-03-4')['Adj Close'] #scope
norm_data.columns = ['GSPC_Adj_Close']
norm_data.info()
norm_data.head(5)


# In[4]:


norm_data_ret = np.log(norm_data/norm_data.shift(1)).dropna()
norm_data_ret.tail(5)


# ## Shapiro–Wilk test

# In[5]:


from scipy.stats import shapiro

shapiro_results = shapiro(norm_data_ret)
print("Shapiro-Wilk test result: ", shapiro_results)

# Get P
p_value = shapiro_results[1]
print("P-value: ", p_value)

#n = 10
#print(format(p_value, '.{}f'.format(n)))


# ## Normal-d fit

# In[6]:


def normal(x, n):
    u = x.mean()
    s = x.std()

    # divide [x.min(), x.max()] by n
    x = np.linspace(x.min(), x.max(), n)

    y = stats.norm.pdf(x, u, s)

    return x, y, x.mean(), x.std()

x, y, u, s = normal(norm_data_ret, 100000)
plt.plot(x, y, linewidth = 2, label='Normal Distribution fit')

plt.hist(norm_data_ret, bins = 1000, density = True, alpha = 0.6, color = 'g', label = 'Data') #fit model

title = "Normal Distribution S&P 500 Fitting" 
plt.title(title)
plt.legend()
plt.show()


# # Whole Data Modeling Part (Yuhan)

# In[7]:


tickers_list = ['^GSPC']
t_data = yf.download(tickers_list,start = '2000-01-01', end = '2022-01-1')['Adj Close'] #scope
t_data.columns = ['GSPC_Adj_Close']
t_data.info()
t_data.head(5)
t_data_ret = np.log(t_data/t_data.shift(1)).dropna()
t_data_ret.tail(5)


# ## T-distribution

# In[8]:


from scipy.stats import t

def td(x, n):
    dof, mu, std = t.fit(x)
    xmin, xmax = plt.xlim([-0.1, 0.1])
    x = np.linspace(xmin, xmax, n)
    y = t.pdf(x, dof, mu, std)
    
    return dof, mu, std, x, y

dof, mu, std, x, y = td(t_data_ret, 1000)
x1, y1, u, s = normal(norm_data_ret, 100000)

plt.plot(x1, y1, linewidth = 2, label='Normal Distribution fit', color = 'orange')
plt.hist(t_data_ret, bins=100, density = True, alpha = 0.6, color = 'g', label = 'S&P 500 Log Returns' )
plt.plot(x, y, linewidth = 2, label='T-distribution fit')

title = "Student-t Distribution S&P 500 Fitting" 
plt.title(title)
plt.legend()
plt.show()


# ## T-distribution QQ-plot

# In[9]:


def tqq(x):
    fig = sm.qqplot(t_data_ret, stats.t, distargs=(dof,), loc=mu, scale=std, line = '45', color ='g')
    title = 't-distribution Probability Plot'
    plt.title(title)
    return plt.show()

tqq(t_data_ret)


# ## Normal-distribution QQ-plot

# In[10]:


def normqq(x):
    sm.qqplot(x, line='s', color='g')
    title = 'Normal distribution Probability Plot'
    plt.title(title)
    return plt.show()

normqq(t_data_ret)


# In[ ]:





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 24 10:36:06 2022

@author: fu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from arch.__future__ import reindexing
import seaborn as sns


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


data = pd.read_csv('S&P500.csv', index_col=(0))
data = data['Price']
data.describe()
returns = 100 * data.pct_change()
returns.fillna(0,inplace=True)
returns.describe()

# stationarity test
returns.plot() 
result = adfuller(returns, autolag = 'AIC') # pass the difference test
result
returns.diff(1)
plot_acf(returns) 
plot_pacf(returns)

# white noise test
LjungBox = acorr_ljungbox(returns,lags = 30)
LjungBox # p value < 0.05, not white noise

# calculate true volatility
vol = returns.std()

# rolling model predicting
rolling_predictions = []
test_size = 365

for i in range(test_size):
    train = returns[ : -(test_size-i)]
    model = arch_model(train, p = 1, q = 1)
    model_fit = model.fit(disp = 'off')
    pred = model_fit.forecast(horizon = 1)
    rolling_predictions.append(np.sqrt (pred.variance.values[-1,:][0]))

rolling_predictions = pd.Series(rolling_predictions, index = returns.index[-365:])

# drawing
fig,ax = plt.subplots(figsize=(10,4))
ax.spines[['top','right']].set_visible(False)
plt.plot(rolling_predictions)
plt.title('Rolling Prediction')

# compare with true value(1 year data)
fig,ax = plt.subplots(figsize=(13,4))
ax.grid(which="major", axis='y', alpha=0.3, zorder=1)
ax.spines[['top','right']].set_visible(False)
plt.plot(returns[-365:])
plt.plot(rolling_predictions)
plt.title('S&P 500 Volatility Prediction - Rolling Forecast')
plt.legend(['True Daily Returns', 'Predicted Volatility'])

sns.distplot(rolling_predictions, color="b", bins=10, kde=True)

# whole data model predicting
train_t = arch_model(returns[-365:],mean='AR',lags=10,vol='GARCH') 
model_fit_t = train_t.fit()
model_fit_t.summary()
#pred2 = model_fit_t.forecast()
#model_fit_t.plot()
model_fit_t.resid.plot(figsize=(12,5))
plt.title('GARCH(1,1) residual ',size=15)
plt.show()
model_fit_t.conditional_volatility.plot(figsize=(12,5),color='r')
plt.title('conditional residual',size=15)
plt.show()












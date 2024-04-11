import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from scipy import stats
from scipy.stats import genextreme as gev
from scipy.stats import pareto

price = yf.download(['^GSPC'], 
                    start='2009-3-4', 
                    end='2019-3-4')['Adj Close']

# try 2008-03-04 ~ 2009-03-04 to see there are too many extreme values.

re = price.pct_change()[1:]
plt.hist(re, bins=200, density=1, alpha=0.7, label='Data')


re_list = re.tolist()
s = pd.DataFrame(re_list,columns = ['re'])

### GEV simulation ###

fit = gev.fit(re) 
# shape, loc, scale
# shape:xi > 0, we got Frechet distribution

x = np.linspace(-0.05, 0.05, 200)
y = gev.pdf(x, *fit)
plt.plot(x, y, label='GEV Distribution fit')
plt.title('GEV Distribution S&P 500 Fitting')
plt.legend(loc='upper right')


# u = s['re'].mean()  # mean
# std = s['re'].std()  # std



import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import yfinance as yf
import _datetime as dt
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


yf.pdr_override()

start = dt.datetime(2020, 1, 1)

tickers = ['BBAS3.SA', 'BRAX11.SA']
prices = pd.DataFrame()

for t in tickers:
    prices[t] = wb.get_data_yahoo(t, start=start)['Adj Close']

log_retruns = np.log(prices/prices.shift(1))
log_retruns = log_retruns.drop(log_retruns.index[0])

cov = log_retruns.cov() * 250
cov_with_market = cov.iloc[0, 1]

bbas_returns = log_retruns['BBAS3.SA']
market_returns = log_retruns['BRAX11.SA']

market_var = market_returns.var() * 250

#Regressão
x1 = sm.add_constant(market_returns)
reg = sm.OLS(bbas_returns, x1).fit()

#print(reg.summary())

#Estimadores
beta, alfa, r_valor, p_valor, std_err = stats.linregress(market_returns, bbas_returns)

#print(beta)

plt.scatter(market_returns, bbas_returns, label='Dados')
plt.plot(market_returns, alfa + beta * market_returns, color='red', label=f'Regressão Linear\nBeta: {beta:.2f}')
plt.xlabel('Retornos do Mercado')
plt.ylabel('Retornos do BBAS3')
plt.title('Beta de Banco do Brasil')
plt.legend()
plt.grid(True)
plt.savefig(f'Beta de BBAS3')
plt.show()

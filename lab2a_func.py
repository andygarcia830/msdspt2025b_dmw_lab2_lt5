import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.ticker as tck
from sklearn.decomposition import PCA


# Downloads 1 year worth of price data
def download(symbol, start=None, days=None):
    end = None
    if not start:
        start = datetime.datetime(2000,1,1)
    if days:
        delta = datetime.timedelta(days=days)
        end = start + delta
    return yf.download(symbol, start=start, end=end)['Adj Close']

def load_data(tickers, start, filename, days = None, force=False):
    if not force:
        try:
            df_spx = pd.read_csv(filename, index_col=0)
            return df_spx
        except FileNotFoundError:
            pass
    df_spx = pd.DataFrame()
    for item in tickers['Symbol']:
        stock = download(item, start, days=days)
        df_spx = pd.concat((df_spx, stock.rename(item)), axis=1)
    df_spx.to_csv(filename)
    return df_spx

def clean_data(df):
    df = df.reset_index(drop=True)
    df.fillna(0, inplace=True)
    df = df.loc[(df!=0).any(axis=1)]
    df = df.loc[:, (df != 0).all(axis=0)]
    return df

def calculate_returns(df):
    R = df.rolling(window=2).apply(lambda x: (x.iloc[1] - x.iloc[0])/x.iloc[0]).fillna(0).reset_index(drop=True)
    Y = ((R - R.mean(axis=0))/R.std(axis=0)).fillna(0)
    return R, Y

def consrtuct_correlation_matrix(Y, filename_corr, force = False):
    if not force:
        try:
            return pd.read_csv(filename_corr, index_col=0)
        except FileNotFoundError:
            pass
    p = pd.DataFrame(np.corrcoef(Y.T))
    p.to_csv(filename_corr)
    return p

def calculate_pca(p):
    pca = PCA()
    pca.fit(p)
    p = np.float64(p)
    w_prime = pca.components_
    variance_explained = pca.explained_variance_
    l = pca.singular_values_
    for i in range(len(w_prime)):
        w_prime[i] = w_prime[i]/sum(w_prime[i])
    return w_prime, variance_explained, l

def calculate_eigen_returns(w_prime, R):
    F = pd.DataFrame(w_prime)
    F = F.divide(R.std(axis=0).to_numpy(), axis = 1)
    F = F @ R.T.to_numpy()   
    F = F.T
    return F

def plot_returns(year, F, R, ax):
    ax.plot(F[0].cumsum(),  label="Market Eigenportfolio Returns")
    ax.plot(R.cumsum(), label = "Market Cap Weighted Actual Returns")
    ax.set_ylabel(year)
    ax.legend()
    
def get_market_returns(startdate, days=None):
    spx_market = download('^SPX', startdate,days=days)
    R_market = spx_market.rolling(window=2).apply(lambda x: (x.iloc[1] - x.iloc[0])/x.iloc[0]).fillna(0).reset_index(drop=True)
    R_market = R_market.div(R_market.std(axis=0)).fillna(0)
    return R_market

def plot_eigenportfolio(w_prime, tickers, E, R, n):
    E = pd.DataFrame(w_prime)
    E.columns = R.columns
    E = E.sort_values(n,axis=1, ascending=False)
    fig, ax = plt.subplots(1,1, figsize=(15,5), layout='constrained')
    ax.grid(visible=True,which='major')
    major_ticks_x = [x for x in range(len(E.columns))][::5]
    major_labels_x = E.columns[::5]
    ax.set_xticks(major_ticks_x)
    ax.set_xticklabels(major_labels_x, rotation=90)
    ax.plot(E.columns,E.iloc[n])
    ax.set_xlabel(f'Eigenvector {n}')
    plt.hlines(0,0,len(E.columns), colors='black')
    print('Top 10 Stocks')
    list10 = tickers.loc[tickers['Symbol'].isin(E.columns[:10])][['Symbol','Security','GICS Sector']]
    print(list10)

    print('\nBottom 10 Stocks')
    list10 = tickers.loc[tickers['Symbol'].isin(E.columns[-10:])][['Symbol','Security','GICS Sector']]
    print(list10)

def plot_eigen_returns(F, ax):
    ax.plot(F[0].cumsum(),  label="Market Eigenportfolio Returns")
    for i in range(1,11):
        ax.plot(F[i].cumsum(),  label=f"Eigenportfolio {i+1} Returns")
    plt.legend()
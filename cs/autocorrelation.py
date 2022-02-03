# -*- coding: utf-8 -*-
"""
Created on Wed May 19 15:26:44 2021

@author: domin
"""

from statsmodels.graphics import tsaplots
import matplotlib.pyplot as plt
import numpy as np

def autocorr(df,col_index,days_lag=5):
    # Display the autocorrelation plot of your time series
    fig = tsaplots.plot_acf(df.iloc[:,col_index],lags=days_lag*24,color='palevioletred')
    '''
    lag= time distance, so 1 = 1 sampling unit -> here: 1 hour. 
    We should the correlation peaks each 24hours if there is a circadian periodicity.
    CI placed at 2*standard error (according to Github source code.) 
    If the autocorrelation coef is in the CI, it is regarded as not significant.
    '''
    ticks=(np.arange(0,days_lag+1,step=1))
    locs=(np.linspace(0,days_lag*24,num=days_lag+1,endpoint=True))
    plt.xticks(locs,ticks)
    plt.show()
    
def pautocorr(df,col_index,days_lag=5,alpha=0.01):
    # Display the autocorrelation plot of your time series
    fig = tsaplots.plot_pacf(df.iloc[:,col_index],lags=days_lag*24,color='palevioletred',zero=False,alpha=alpha)
    '''
    lag= time distance, so 1 = 1 sampling unit -> here: 1 hour. 
    We should the correlation peaks each 24hours if there is a circadian periodicity.
    CI placed at 2*standard error (according to Github source code.) 
    If the autocorrelation coef is in the CI, it is regarded as not significant.
    '''
    ticks=(np.arange(0,days_lag+1,step=1))
    locs=(np.linspace(0,days_lag*24,num=days_lag+1,endpoint=True))
    plt.xticks(locs,ticks)
    plt.ylim(-0.1,0.2)
    plt.show()
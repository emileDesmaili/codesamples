# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:49:42 2021

@author: eesmaili
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns
import math
from pandas import DataFrame
from openpyxl import load_workbook
import scipy.stats as stats
from sklearn.decomposition import PCA   
import scipy as sp
#from filterpy.kalman import KalmanFilter
from pykalman import KalmanFilter, UnscentedKalmanFilter

from scipy import linalg
from matplotlib import pyplot as plt
plt.style.use('default')
plt.style.use('ggplot')

from numpy import linalg 
from scipy.linalg import sqrtm
import pyflux as pf


import statsmodels.api as sm


"-------------------------------------"

path_w=r'P:\Services\Fmr\forex\emile\Kalman hedging\equity hedge\equity_weekly_reverse.xlsm'
path_m=r'P:\Services\Fmr\forex\emile\Kalman hedging\equity hedge\equity_monthly_reverse.xlsm'


book = load_workbook(path)
writer = pd.ExcelWriter(path, engine = 'openpyxl')
writer.book = book


   
"---------------------------------------"


def kalman_equity(N,episode,freq):
    
    if freq =="Weekly":
        path=path_w
    else:
        path=path_m
        
 
    data=pd.read_excel(path,
                      sheet_name=episode,
                      header=0,index_col=0,parse_dates=[0]
                    )
    vol_data=pd.read_excel(path,
                      sheet_name='vol',
                      header=0,index_col=0,parse_dates=[0]
                    )
    data.dropna(inplace=True)
    target=data.columns[:-12]

    Y=DataFrame(data[target])
    X=data.drop(target,axis=1)

    X_orth=DataFrame(np.dot(X,torsion(np.cov(X,rowvar=0),model='minimum-torsion').T),
                 index=X.index,columns=X.columns)

#mod = sm.OLS(Y['MXN'],X)
#res = mod.fit(cov_type='HAC',cov_kwds={'maxlags':1})
#print(res.summary())

    delta = 1e2*0.999999999998
     

    obs_mat = np.vstack([X['MXN'],X['GBP'],X['AUD'],X['CAD'],X['NOK'],
                         X['SEK'],X['CHF'],X['JPY'],X['EUR'],X['PLN'],
                         X['ZAR'],X['NZD'],np.ones((len(X),1))
                     ]).T[:, np.newaxis]

    n=obs_mat[:,0,:].shape[1]
    trans_cov = delta / (1 - delta) * np.eye(n)
    beta=[]

    

    for label, content in Y.items():

        kf = KalmanFilter(n_dim_obs=1, n_dim_state=n,
                          transition_matrices=np.eye(n),
                          initial_state_mean=np.zeros(n),
                          initial_state_covariance=np.eye(n)*100000,
                          observation_matrices=obs_mat)
    
        state_means, state_covs = kf.em(Y[label].values,em_vars=['transition_covariance',                     
                    'observation_covariance','initial_state_mean',
                    'initial_state_covariance']).filter(Y[label].values)

        beta.append(state_means)

    
    beta=np.array(beta)


    beta1=DataFrame(beta[0],index=X.index)
    beta2=DataFrame(beta[1],index=X.index)
    beta3=DataFrame(beta[2],index=X.index)
    beta4=DataFrame(beta[3],index=X.index)
    
    
    
    
    w_star=pd.concat([beta1,beta2,beta3,beta4],axis=1)




    
    filename="%s.xlsx" % freq
    
    dir_name=r'P:\Services\Fmr\forex\emile\Kalman hedging\equity hedge'
    
    pathos=os.path.join(dir_name, filename)

    mywriter = pd.ExcelWriter(pathos,
                          engine='xlsxwriter')   
    workbook=mywriter.book
    worksheet=workbook.add_worksheet('w_star')
    mywriter.sheets['w_star'] = worksheet
    w_star.to_excel(mywriter,sheet_name='w_star',startrow=0 , startcol=0)  

    mywriter.save()

"-----------------------------------------------------"    
    
kalman_equity(6,"S&P","Monthly")

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:12:29 2020

@author: eesmaili
"""

import pandas as pd
import numpy as np
import seaborn as sns
import math
from pandas import DataFrame
from openpyxl import load_workbook
from sklearn.covariance import LedoitWolf
import scipy.stats as stats
from sklearn.decomposition import PCA   
import scipy as sp
from scipy.spatial.distance import pdist
from scipy import linalg
from matplotlib import pyplot as plt
plt.style.use('default')
plt.style.use('seaborn-pastel')
from scipy.cluster.hierarchy import linkage
from numpy import linalg 
from scipy.linalg import sqrtm

from sklearn.preprocessing import StandardScaler

"-------------- declar------------------------"

path=r'P:\Services\Fmr\forex\emile\recherches cryptos\assetalloc2.xlsx'
book = load_workbook(path)
writer = pd.ExcelWriter(path, engine = 'openpyxl')
writer.book = book

data=pd.read_excel(path,
                      sheet_name="Sheet1",
                      header=0,index_col=0,parse_dates=[0]
                    )

data_no_crypto=data.iloc[:,:-4]

def pf(df, L):
    """Limit ENTIRE column width (including header)"""
    pd.set_option("display.max_colwidth", L)
    return df.rename(columns=lambda x: x[:L - 3] + '' if len(x) > L else x)
 
data=pf(data,30)
data_no_crypto=pf(data_no_crypto,30)


"-------------------------------------------------------------------------"


def plot_corr():
    path=r'P:\Services\Fmr\forex\emile\recherches cryptos\assetalloc.xlsx'
    data=pd.read_excel(path,
                      sheet_name="Sheet1",
                      header=0,index_col=0,parse_dates=[0]
                    )
    corr=data.pct_change().iloc[1:].corr()
    clustermap=sns.clustermap(corr,method='single',cmap='coolwarm',
                              figsize=(6.5,6.5))
    return clustermap


"-------------------optimizers ---------------------------------------------"
                             
def port_maxsr(mean,cov):
    def objective(W,R,C):
        meanp=np.dot(W.T,mean)
        varp=np.dot(np.dot(W.T,cov),W)
        util=meanp/varp**0.5
        return 1/util
    n=len(cov)
    W=np.ones([n])/n
    b_=[(0.,1.)for i in range(n)]
    c_=({'type':'eq','fun': lambda W: sum(W)-1})
    optimized=sp.optimize.minimize(objective,W,(mean,cov),method='SLSQP',
                         constraints=c_,bounds=b_,
                         options={'maxiter':100,'ftol':1e-12})
    return optimized.x


def port_minvol(cov):
    def objective(W,cov):
        varp=np.dot(np.dot(W.T,cov),W)
        util=varp
        return util
    n=len(cov)
    W=np.ones([n])/n
    b_=[(0.,1.)for i in range(n)]
    c_=({'type':'eq','fun': lambda W: sum(W)-1})
    optimized=sp.optimize.minimize(objective,W,cov,method='SLSQP',
                         constraints=c_,bounds=b_,
                        options={'maxiter':400,'ftol':1e-15} )
    return optimized.x
    
def port_maxret(mean,cov):
    def objective(W,R,C):
        meanp=np.dot(W.T,mean)
        util=1/meanp
        return util
    n=len(cov)
    W=np.ones([n])/n
    b_=[(0.,1.)for i in range(n)]
    c_=({'type':'eq','fun': lambda W: sum(W)-1})
    optimized=sp.optimize.minimize(objective,W,(mean,cov),method='SLSQP',
                         constraints=c_,bounds=b_,
                       options={'maxiter':400,'ftol':1e-15} )
    return optimized.x
    

def port_minvol_ro(mean,cov,ro):
    def objective(W,R,C,ro):
        varp=np.dot(np.dot(W.T,cov),W)
        util=varp
        return util
    n=len(cov)
    W=np.ones([n])/n
    b_=[(0.,1.)for i in range(n)]
    c_=({'type':'eq','fun': lambda W: sum(W)-1},{'type':'eq','fun': lambda W: np.dot(W.T,mean)-ro})
    optimized=sp.optimize.minimize(objective,W,(mean,cov,ro),method='SLSQP',
                          constraints=c_,bounds=b_,
                          options={'maxiter':200,'ftol':1e-15})
    return optimized.x



##### HRP ####
def getIVP(cov,**kargs):
    # Compute the inverse-variance portfolio
    ivp=1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp
#------------------------------------------------------------------------------
def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar
#------------------------------------------------------------------------------
def getQuasiDiag(link):
    # Sort clustered items by distance
    link=link.astype(int)
    sortIx=pd.Series([link[-1,0],link[-1,1]])
    numItems=link[-1,3] # number of original items
    while sortIx.max()>=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
        df0=sortIx[sortIx>=numItems] # find clusters
        i=df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1],index=i+1)
        sortIx=sortIx.append(df0) # item 2
        sortIx=sortIx.sort_index() # re-sort
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()
#------------------------------------------------------------------------------
def getRecBipart(cov,sortIx):
# Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,k in ((0,int(len(i)/2)),\
                (int(len(i)//2) ,len(i))) if len(i)>1] # bi-section
        for i in range(0,len(cItems),2): # parse in pairs
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            cVar0=getClusterVar(cov,cItems0)
            cVar1=getClusterVar(cov,cItems1)
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2
    return w
#------------------------------------------------------------------------------
def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist=((1-corr)*0.5)**.5 # distance matrix
    return dist



def HRP(x):
    cov,corr=x.cov(),x.corr()
    cov=DataFrame(LedoitWolf().fit(x).covariance_,columns=x.columns,
                  index=x.columns)
    d_corr = correlDist(corr)
    Y=pdist(d_corr)
    link = linkage(Y, 'single')
    sortIx = getQuasiDiag(link)
    sortIx=corr.index[sortIx].tolist()
    weights = getRecBipart(cov, sortIx)
    return weights.reindex(index=x.columns)






"----------------- Efficient Frontier ----------------------------------------"

def eff_frontier():
    returns=np.array(data.pct_change().iloc[1:])
    no_crypto=np.array(data_no_crypto.pct_change().iloc[1:])
    mean=np.mean(no_crypto,axis=0)*252
    cov=np.cov(no_crypto,rowvar=0)*252
    meanCrypto=np.mean(returns,axis=0)*252
    covCrypto=np.cov(returns,rowvar=0)*252
    
    n_sim=100
    vol=[]
    vol_crypto=[]
    ro=[]
    w_vec=[]
    w_vec_crypto=[]
    #np.dot(port_maxret(mean,cov),mean.T)
    ro=np.linspace(0.01,0.25,n_sim)
    for r in ro:
        w=np.array(port_minvol_ro(mean,cov,r))
        w_vec.append(w)
        vol=np.append(vol,np.sqrt(np.dot(np.dot(w.T,cov),w)))
        w2=np.array(port_minvol_ro(meanCrypto,covCrypto,r))
        w_vec_crypto.append(w2)
        vol_crypto=np.append(vol_crypto,np.sqrt(np.dot(np.dot(w2.T,covCrypto),
                                                   w2)))
    w_vec=np.array(w_vec)
    w_vec_crypto=np.array(w_vec_crypto)

    
    line2,=plt.plot(vol_crypto,ro,label='with cryptos',color="steelblue",
                linestyle='dashed')
    line1,=plt.plot(vol,ro,label='base case',color="darkviolet",lw=2
                )
    plt.legend(handles=[line1,line2])
    plt.xlabel('realized vol')
    plt.ylabel('mean return')
        
    return plt.show







"---------------------------------------------------------"





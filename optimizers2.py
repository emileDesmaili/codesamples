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
import scipy as sp
from scipy import linalg
from matplotlib import pyplot as plt
plt.style.use('seaborn-pastel')
from numpy import linalg 






"-------------------classical optimizers ---------------------------------------------"
                             
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



#--------------- Equal Risk Contribution (LeastSquares) --------------------------

def erc_w(cov, x0=None, options=None, scale_factor=10000,
                 pcr_tolerance=0.01, ignore_objective=False):

    # check matrix is PD
    np.linalg.cholesky(cov)

    if not options:
        options = {'ftol': 1e-200, 'maxiter': 800}

    def fun(x):
        # these are non normalized risk contributions, i.e. not regularized
        # by total risk, seems to help numerically
        risk_contributions = x.dot(cov) * x
        a = np.reshape(risk_contributions, (len(risk_contributions), 1))
        # broadcasts so you get pairwise differences in risk contributions
        risk_diffs = a - a.transpose()
        sum_risk_diffs_squared = np.sum(np.square(np.ravel(risk_diffs)))

        return sum_risk_diffs_squared / scale_factor

    N = cov.shape[0]
    if x0 is None:
        x0 = 1 / np.sqrt(np.diag(cov))
        x0 = x0 / x0.sum()

    bounds = [(0, 1) for i in range(N)]
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    res = sp.optimize.minimize(fun, x0, method='SLSQP', bounds=bounds,
                                  constraints=constraints,
                                  options=options)
    weights = res.x
    risk_squared = weights.dot(cov).dot(weights)
    pcrs = weights.dot(cov) * weights / risk_squared
    pcrs = np.reshape(pcrs, (len(pcrs), 1))
#    pcr_max_diff = np.max(np.abs(pcrs - pcrs.transpose()))
#    if not res.success:
#        if ignore_objective and (pcr_max_diff < pcr_tolerance):
#            return weights
#        else:
#            msg = ("Max difference in percentage contribution to risk "
#                   "in decimals is {0:.2E}, "
#                   "tolerance is {1:.2E}".format(pcr_max_diff, pcr_tolerance))
#            warnings.warn(msg)
#            raise RuntimeError(res)
#    if pcr_max_diff > pcr_tolerance:
#        raise RuntimeError("Max difference in percentage contribution to risk "
#                           "in decimals is %s which exceeds tolerance of %s." %
#                           (pcr_max_diff, pcr_tolerance))

    return weights



"----------------- Efficient Frontier ----------------------------------------"

def eff_frontier(data,n_sim=100):
    returns=np.array(data.pct_change().iloc[1:])
    mean=np.mean(returns,axis=0)*252
    cov=np.cov(returns,rowvar=0)*252
    vol=[]
    ro=[]
    w_vec=[]
    ro=np.linspace(0.01,0.25,n_sim)
    for r in ro:
        w=np.array(port_minvol_ro(mean,cov,r))
        w_vec.append(w)
        vol=np.append(vol,np.sqrt(np.dot(np.dot(w.T,cov),w)))

    w_vec=np.array(w_vec)
    line1,=plt.plot(vol,ro,label='efficient frontier',color="darkviolet",lw=2)
    plt.xlabel('realized vol')
    plt.ylabel('mean return')
        
    return plt.show


"-------------risk analytics functions----------------------------------------------"
def DD(returns):
    totalreturn=DataFrame(returns).cumsum()
    drawdown = totalreturn - totalreturn.cummax()
    drawdown=returns/returns.cummax()-1
    return drawdown

def CVAR(returns,alpha):
    returns=-returns
    N=len(returns)
    k=math.ceil(N*alpha)
    z=np.sort(returns,axis=0)
    z=z[::-1]
    ES=np.mean(z[0:k],axis=0)
    VAR = z[k]
    return ES,VAR


   
       

   




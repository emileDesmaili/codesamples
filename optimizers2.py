# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:12:29 2020

@author: eesmaili
"""

import numpy as np
import math
import scipy as sp
from scipy import linalg
from matplotlib import pyplot as plt
plt.style.use('seaborn-pastel')
from numpy import linalg 


"-------------------traditional optimizers ---------------------------------------------"
                             
def port_maxsr(mean,cov):
# weights are chosen to maximize the ratio (mean/variance) of the portfolio
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
# weights are chosen to minimize the ratio variance of the portfolio
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
# weights are chosen to maximize the ratio mean return of the portfolio
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
# weights are chosen to minimize the variance of the portfolio given a mean return target
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



"--------------- Equal Risk Contribution (LeastSquares) --------------------------"

def erc_w(cov, x0=None, options=None):
# weights are chosen so that risk contributions of each asset is equal
# the risk contribution of asset i is defined as the derivative of the portfolio
# sqrt(variance) with respect to the weight of asset i
    if not options:
        options = {'ftol': 1e-200, 'maxiter': 800}
    def fun(x):
        # these are non normalized risk contributions, i.e. not regularized
        # by total risk, seems to help numerically
        # the optimizer minimizes the sum of squared pairwise diff. in RCs 
        risk_contributions = x.dot(cov) * x
        a = np.reshape(risk_contributions, (len(risk_contributions), 1))
        # broadcasts so you get pairwise differences in risk contributions
        risk_diffs = a - a.transpose()
        sum_risk_diffs_squared = np.sum(np.square(np.ravel(risk_diffs)))
        return sum_risk_diffs_squared
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
    return weights


"----------------- Efficient Frontier ----------------------------------------"
def eff_frontier(data,n_sim=100):
# Function to plot the frontier of minimum variance portfolios given a return target ro
# Data needs to be a (T,N) matrix of N asset prices over T days
    returns=np.array(data.pct_change().iloc[1:])
    mean=np.mean(returns,axis=0)*252
    cov=np.cov(returns,rowvar=0)*252
    vol=[], ro=[], w_vec=[]
    # list of returns ro to iterate through to compute the corresponding min.variance portfolio
    ro=np.linspace(0.01,0.25,n_sim)
    for r in ro:
        w=np.array(port_minvol_ro(mean,cov,r))
        w_vec.append(w)
        vol=np.append(vol,np.sqrt(np.dot(np.dot(w.T,cov),w)))
    #plot
    w_vec=np.array(w_vec)
    plt.plot(vol,ro,label='efficient frontier',color="darkviolet",lw=2)
    plt.xlabel('realized vol')
    plt.ylabel('mean return')      
    return plt.show


"-------------risk analytics functions----------------------------------------------"
def DD(returns):
 # computes the maximum drawdown of the portfolio
    totalreturn=DataFrame(returns).cumsum()
    drawdown = totalreturn - totalreturn.cummax()
    drawdown=returns/returns.cummax()-1
    return drawdown 

def CVAR(returns,alpha):
# computes the Value at Risk (maximum loss given a confidence lvel alpha
# and the Expected Shortfall(expected loss conditional to being beyond the VaR)
    returns=-returns
    N=len(returns)
    k=math.ceil(N*alpha)
    z=np.sort(returns,axis=0)
    z=z[::-1]
    ES=np.mean(z[0:k],axis=0)
    VAR = z[k]
    return ES,VAR


   
       

   




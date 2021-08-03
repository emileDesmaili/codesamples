# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:24:34 2021

@author: eesmaili
"""
# this script:
# imports  market data of interest rate curves of countries 
# reconstructs the curves by performing PCA on the first 3 PCs
# returns the residuals' times series (for statistical abritrage) in an excel file

# import librairies
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from sklearn.decomposition import PCA   
import os

# load the data from the xlsm file

path=r'P:\Services\Fmr\forex\EM Bonds tools\PCA\dashboard_data.xlsm'
book = load_workbook(path)
writer = pd.ExcelWriter(path, engine = 'openpyxl')
writer.book = book

# function to perform PCA for one rate surface: a (T,N) matrix with T days
# and N curve maturity points
def my_PCA(country): 
# data is the (T,N) matrix for one country/currency
   data=pd.read_excel(path,
                      sheet_name=country,
                      header=0,index_col=0,parse_dates=[0]
                    )
# fit a PCA with first 3 components of the curve: level, slope and concavity
   pca = PCA(3)
   pca.fit(data)
# reconstruct the curve with these 3 components and take the residual to assess
# the abritrage opportunity
   scores = pca.transform(data)
   reconstruct = pca.inverse_transform(scores)
   residuals=data-reconstruct
# as additional information, the explained variance ratio of each PC 
   print(np.round(pca.explained_variance_ratio_,3))
   return residuals*100

# declare the country codes of local currency interets rates to be used
# as they are on the .xlsm file

my_set=['AUD','CZK','THB','KRW','HUF','PLN','NZD','GBP','CAD','JPY','CHF','COP',
         'MXN','RUB','ZAR','BRL','TRY','USD','EUR']

# function to create an excel file with the output of each PCA reconstruction
def excelize(my_set=my_set):
    for label in my_set:
        print(label)
        filename="%s.xlsx" % label
        dir_name=r'P:\Services\Fmr\forex\EM Bonds tools\PCA\Python out'
        pathos=os.path.join(dir_name, filename)
        mywriter = pd.ExcelWriter(pathos,
                          engine='xlsxwriter')   
        workbook=mywriter.book
        worksheet=workbook.add_worksheet('pca')
        mywriter.sheets['pca'] = worksheet
        my_PCA(label).to_excel(mywriter,sheet_name='pca',startrow=0 , startcol=0)  
        mywriter.save()
        
# perform the operation on the set of currencies    
excelize()
    


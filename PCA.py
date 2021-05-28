# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:24:34 2021

@author: eesmaili
"""


import pandas as pd
import numpy as np
import seaborn as sns

from pandas import DataFrame
from openpyxl import load_workbook

from sklearn.decomposition import PCA   
import scipy as sp

from matplotlib import pyplot as plt
import os
plt.style.use('default')
plt.style.use('seaborn-pastel')




path=r'P:\Services\Fmr\forex\EM Bonds tools\PCA\dashboard_data.xlsm'
book = load_workbook(path)
writer = pd.ExcelWriter(path, engine = 'openpyxl')
writer.book = book

def my_PCA(country):
    
   data=pd.read_excel(path,
                      sheet_name=country,
                      header=0,index_col=0,parse_dates=[0]
                    )
   pca = PCA(3)
   pca.fit(data)
   scores = pca.transform(data)
   reconstruct = pca.inverse_transform(scores)
   residuals=data-reconstruct

   print(np.round(pca.explained_variance_ratio_,3))
   return residuals*100



my_set=['AUD','CZK','THB','KRW','HUF','PLN','NZD','GBP','CAD','JPY','CHF','COP',
         'MXN','RUB','ZAR','BRL','TRY','USD','EUR']

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
    
excelize()
    


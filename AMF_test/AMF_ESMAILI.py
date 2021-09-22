# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 14:03:34 2021

@author: emile
"""
import pandas as pd
import numpy as np


# import des csv
classe_liquidite = pd.read_csv(r'C:\Users\emile\Documents\Python Scripts\AMF\Classe_liquidite.csv',
                               sep=';')

# j'ai remarqué en regroupant qu'un ou deux ocntrat etait labelisé "Not liquid" et non Not Liquid")
classe_liquidite['Liquidity_Status'].str.replace('Not liquid','Not Liquid')

data_volumes = pd.read_csv(r'C:\Users\emile\Documents\Python Scripts\AMF\Data_volumes.csv',
                           sep=';', decimal=',', parse_dates=[0])

titres_contrat_liquidite = pd.read_csv(r'C:\Users\emile\Documents\Python Scripts\AMF\Titres_contrat_liquidite.csv',
                                       sep=';')

# calcul du volume par jour, j'utilise set() pour obtenir le nombre de jours différents

volume_par_jour= data_volumes['Volume_Traded'].sum()/len(set(data_volumes['Date']))
print('le volume par jour est: ', np.round(volume_par_jour))

# calcul des 10 titres les plus echangés

isin_df= data_volumes.groupby(["IsinLabel"])['Volume_Traded'].sum()
print("Les 10 titres les plus échangés: \n", isin_df.nlargest(10))
isin_df.index.str.replace(',','.')

# calcul de la plus forte variation intraday

data_volumes['variation_intraday']=data_volumes['HighPrice']-data_volumes['LowPrice']
print("plus forte variation intraday: \n", data_volumes['variation_intraday'].nlargest(1))

# calcul du volume par jour selon la classe de liquidité

liquidity_status_df=data_volumes.merge(classe_liquidite, left_on='Isin',
                             right_on='Isin').groupby(['Liquidity_Status'])['Volume_Traded'].sum()
liquidity_status_df=liquidity_status_df/len(set(data_volumes['Date']))

 
liquidity_status_df.plot.bar(ylabel='volumes')

# dataframe avec le statut contrat de liquidité puis calcul du volume par classe

data_volumes['contrat status']=np.where(data_volumes['IsinLabel'].isin(titres_contrat_liquidite['Isin_Label']),
                                     'contrat','no contrat')
contrat_df= data_volumes.groupby(["contrat status"])['Volume_Traded'].sum()
contrat_df=contrat_df/len(set(data_volumes['Date']))

contrat_df.plot.bar(ylabel='volumes')







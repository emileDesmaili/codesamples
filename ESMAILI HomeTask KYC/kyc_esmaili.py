# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 13:03:58 2021

@author: emile
"""


import pandas as pd # data processing 

import statsmodels.api as sm #regression models
import numpy as np


# importing the data and dropping duplicates

df_docs = pd.read_csv(r'C:\Users\emile\Documents\Python Scripts\revolut\RelevantFiles\doc_reports.csv')
df_docs=df_docs.rename(columns={'result': 'result_doc'})
df_docs=df_docs.drop(['Unnamed: 0'],axis=1)
df_face = pd.read_csv(r'C:\Users\emile\Documents\Python Scripts\revolut\RelevantFiles\facial_similarity_reports.csv')
df_face=df_face.rename(columns={'result': 'result_face'})
df_face=df_face.drop(['created_at','user_id','attempt_id','properties','Unnamed: 0'],axis=1)

# building a daily dataframe with the pass rate, doc result rate and face result rate
# the new dataframes columns are aded and do not replaces the original ones on purpose

df = pd.concat([df_docs,df_face],axis=1)
df.index = pd.to_datetime(df['created_at'],format='%Y-%m-%dT%H:%M:%SZ')
df=df.sort_index(axis=0)

face_result = df['result_face'] == "clear"
df['face_result'] = df['result_face'] == "clear"
docs_result = df['result_doc'] == "clear"
df['docs_result'] = df["result_doc"] == "clear"
df['Pass'] = face_result & docs_result

daily_df=df.groupby(pd.Grouper(freq='D')).mean().dropna()
daily_df.plot()

# we will focus on the docs_result dataframe which seems to be the culprit

df_docs.index = pd.to_datetime(df_docs['created_at'],format='%Y-%m-%dT%H:%M:%SZ')
df_docs=df_docs.sort_index(axis=0)
df_docs['doc_result']= df_docs["result_doc"] == "clear"
df_docs['visual_auth']= df_docs["visual_authenticity_result"] == "clear"
df_docs['image_integrity']= df_docs["image_integrity_result"] == "clear"
df_docs['police']= df_docs["police_record_result"] == "clear"

daily_df_docs=df_docs.groupby(pd.Grouper(freq='D')).mean().dropna()
daily_df_docs.plot()

# sensitivity check with OLS regression on daily averages
mod = sm.OLS(daily_df_docs.iloc[:,0], daily_df_docs.iloc[:,1:])
res = mod.fit()
print(res.summary())

# now we will focus on the image integrity components

df_image_integrity=pd.DataFrame()
df_image_integrity['image_integrity']=df_docs["image_integrity_result"] == "clear"
df_image_integrity['support_doc']=df_docs["supported_document_result"] == "clear"
df_image_integrity['image_quality']=df_docs["image_quality_result"] == "clear"
df_image_integrity['color_picture']=df_docs["colour_picture_result"] == "clear"
df_image_integrity['conclusive_doc_quality']=df_docs["conclusive_document_quality_result"] == "clear"

daily_df_image_integrity=df_image_integrity.groupby(pd.Grouper(freq='D')).mean().dropna()
daily_df_image_integrity.plot()

# sensitivity check with OLS regression on daily averages
mod2 = sm.OLS(daily_df_image_integrity.iloc[:,0], daily_df_image_integrity.iloc[:,1:])
res2 = mod2.fit()
print(res2.summary())

# sensitivity check with logit regresion on binary encoded dataframes with
# all intraday values 

encoded_df=pd.DataFrame(np.where(df_docs.drop(['user_id','created_at','attempt_id',
                                            'properties','sub_result','doc_result',
                         'visual_auth','image_integrity','police'],axis=1)=='clear',1,0),
                     index=df_docs.index,columns=df_docs.drop(['user_id','created_at','attempt_id',
                                            'properties','sub_result','doc_result',
                         'visual_auth','image_integrity','police'],axis=1).columns)

mod3=sm.Logit(encoded_df['result_doc'],
                                      encoded_df[['visual_authenticity_result',
                                                     'image_integrity_result',
                                                     'police_record_result']]).fit()
mod3.summary()

mod4=sm.Logit(encoded_df['image_integrity_result'],
                                      encoded_df[['supported_document_result',
                                                     'image_quality_result',
                                                     'colour_picture_result',
                                                     'conclusive_document_quality_result']]).fit()
mod4.summary()

# looking at subresults for CDQR

sub_df=pd.DataFrame()
sub_df2=pd.DataFrame()

# df when CDQR fails
sub_df = df_docs.loc[df_docs['conclusive_document_quality_result'] != 'clear']
sub_df_group=sub_df.groupby(['sub_result']).count()
sub_df_group=sub_df_group/np.sum(sub_df_group)

# df when image quality fails
sub_df2 = df_docs.loc[df_docs['image_quality_result'] != 'clear']
sub_df2_group=sub_df2.groupby(['sub_result']).count()
sub_df2_group=sub_df2_group/np.sum(sub_df2_group)

# plot
pd.concat([sub_df_group[['conclusive_document_quality_result']],
           sub_df2_group[['image_quality_result']]],axis=1).plot.bar()

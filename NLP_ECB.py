# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:38:08 2021

@author: eesmaili
"""


import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import date,timedelta,datetime
import datetime
from textblob import TextBlob
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

from PIL import Image
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


path=r'P:\Services\Fmr\forex\emile\NLP\all_ECB_speeches.csv'

data= pd.read_csv(path, sep='|',encoding= 'utf-8',index_col=0,parse_dates=[0])
analyser = SentimentIntensityAnalyzer()

data=data.dropna(axis=0)

score=[]



for index, row in data.iterrows():

    speech=row['contents']
    soup = BeautifulSoup(speech,"lxml") 
    vader_pos=analyser.polarity_scores(soup.get_text())['pos']
    vader_neg=analyser.polarity_scores(soup.get_text())['neg']
    try:
        compound=vader_pos/(vader_pos+vader_neg)
    except ZeroDivisionError:
        compound = 0    
    score.append((compound))

sentiment=DataFrame(score,index=data.index)
sentiment.rolling(90).mean().plot()


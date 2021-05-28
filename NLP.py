# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 21:39:11 2020

@author: eesmaili
"""

import eikon as ek
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import date,timedelta,datetime
import datetime
from textblob import TextBlob
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS
ek.set_app_key('f07700af22c14ab08d94558231df0a9ad08739ae')
from PIL import Image
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

"---------------------------------------------------------------------------"
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
    
def transform_format(val):
    if val.all() == 0:
        return 255
    else:
        return val
        
def get_logo():
    logo = np.array(Image.open(r'P:\Services\Fmr\forex\emile\recherches cryptos\logos\logo1.png'))

    zelogo = np.ndarray((logo.shape[0],logo.shape[1]), np.int32)
    for i in range(len(logo)):
        zelogo[i] = list(map(transform_format, logo[i]))
    return zelogo
"---------------------------------------------------------------------------"

def daily_sentiment(start):
    analyser = SentimentIntensityAnalyzer()
    try:
        headlines = ek.get_news_headlines('Topic:BRXT AND Language:LEN',100
                                      ,
                                      date_from=start,date_to=start+timedelta(days=15))
    except (TypeError,KeyError):
        return 0
    score=[]
    if headlines.empty:
        return 0
    else:
        for index, headline_row in headlines.iterrows():
            story=headline_row['text']
#        for i in range (headlines.shape[0]):
#            story_id = headlines.iat[i,2]
            try:
#                story=ek.get_news_story(story_id)
                soup = BeautifulSoup(story,"lxml") 
                blob=TextBlob(soup.get_text()).sentiment[0]
                vader=analyser.polarity_scores(soup.get_text())['compound']
                score.append((vader+blob)*0.5)
            except (TypeError,KeyError):
                score.append(0)
                continue
    return np.mean(score)


def sentiment(start):
    sentall=[]
    dates=[]
    start= datetime.datetime.strptime(start, '%Y-%m-%d')
    end=datetime.datetime.now()   
    for s in daterange(start, end):
        print(s)
        dates.append(s)
        sentall.append(daily_sentiment(s))
    return DataFrame(sentall,index=dates)



def wordclouds(start):
      
    text=[]
    start= datetime.datetime.strptime(start, '%Y-%m-%d')
    headlines = ek.get_news_headlines('digital currency AND Language:LEN',
                                      100,
                                      date_from=start)
    if headlines.empty:
        text.append('')
    else:
        for i in range (headlines.shape[0]):
            story_id = headlines.iat[i,2]
            try:
                story=ek.get_news_story(story_id)
                soup = BeautifulSoup(story,"lxml") 
                text.append(soup.get_text())
            except (TypeError,KeyError):
                continue

    stopwords= set(STOPWORDS)
    stopwords.update(["will", "said", "important", "inherit", "color","Galaxy",
                      "Company"])
    wordcloud = WordCloud(stopwords=stopwords,
                          background_color='white').generate(', '.join(text))
    plt.figure(figsize=[10,10])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return plt.show()


    
"-----------------------------------------------------------------------------"

#zelogo=get_logo()
#
#wordclouds('2020-01-01')


reuters_sentiment=sentiment(start='2020-01-01')
reuters_sentiment=reuters_sentiment.rename(columns={0: 'sentiment'})

fx=ek.get_timeseries(['EURGBP='], 
                  start_date='2020-01-01', 
                  end_date=datetime.datetime.now(), 
                  interval='daily')['CLOSE']


#fig, ax = plt.subplots()
#ax2 = ax.twinx()
#reuters_sentiment.rolling(90).mean().plot(ax=ax,label='sentiment',color='lightcoral')
#fx.pct_change().iloc[1:].rolling(90).mean().plot(ax=ax2,label='price')
#ax.legend(['sentiment'],loc='lower right')
#ax2.legend(['price'])
#


fx_ret=fx.pct_change().iloc[1:]

reuters_diff=reuters_sentiment.diff().iloc[1:]
granger_df=pd.concat([reuters_diff,fx_ret],axis=1).dropna()
granger_df2=pd.concat([fx_ret,reuters_diff],axis=1).dropna()



print('---------h0 a: price does not cause sentiment-------')
gc_res = grangercausalitytests(granger_df, 5)
print('---------h0 b: sentiment does not cause price-------')
gc_res2 = grangercausalitytests(granger_df2, 5)

granger_df.plot.scatter('sentiment','CLOSE')
VARmodel=VAR(granger_df)
results=VARmodel.fit(3)
irf = results.irf(10)
irf.plot()


TextBlob("UK growth slows down").sentiment
analyser=SentimentIntensityAnalyzer()
analyser.polarity_scores("UK growth slows down ")

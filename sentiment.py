<<<<<<< HEAD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb
from textblob import TextBlob

def get_sentiment(df):
    df['title_des'] = df['title'] + df['description']
    df = df.loc[:,['item_id','title_des']]
    dict = df.to_dict('records')
    for row in dict:
        temp = TextBlob(' '.join(row['title_des'].values.astype('str')))
        score = temp.sentiment.polarity
        row['sentiment'] = score
    dd = pd.DataFrame(dict)
    return dd
=======
import numpy as np
import pandas as pd
from textblob import TextBlob
import re

def get_sentiment(df):
    df['title_des'] = df['title'] + df['description']
    df = df.loc[:,['item_id','title_des']]
    dict = df.to_dict('records')
    for row in dict:
        try:
            temp = TextBlob(row['title_des'])
            score = temp.sentiment.polarity
        except:
            score = None
        row['sentiment'] = score
    dd = pd.DataFrame(dict)
    dd = dd.drop(['title_des'],axis=0)
    return dd
df_train=pd.read_csv('train.csv',index_col=None)
df1=get_sentiment(df_train)
df1.to_csv('train_sentiment.csv',index=False)
df_test=pd.read_csv('test.csv',index_col=None)
df2=get_sentiment(df_test)
df2.to_csv('test_sentiment.csv',index=False)
>>>>>>> ef102ee4235bbbbd65682fb8672ab89bb21df872

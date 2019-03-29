# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:51:36 2019

@author: swooty
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv('data2.csv')

about_list = df['About'][0:200]

#about_list = df_clean['About']

stop_words = ['an','any','are','as','at','and','be','by','for','has','have','if',
              'in','is','it','its','no','of','on','or','our','that','the',
              'their','this','to','us','was','we','where','which','with',
              'you','your']
vectorizer = CountVectorizer(max_features = 2500, max_df = 0.9, min_df = 1.0/np.float(len(about_list)),stop_words=stop_words)
X = vectorizer.fit_transform(about_list)
wordlist = vectorizer.get_feature_names()
print wordlist

np.savez('featureMatrixSmall', X=X.todense(), about_list= about_list, 
         df_indeces = about_list.index, wordlist=wordlist)

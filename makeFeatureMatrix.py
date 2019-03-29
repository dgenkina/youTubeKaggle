# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:51:36 2019

@author: swooty
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
import re

df = pd.read_csv('data2.csv')
df = df[:][np.logical_not(pd.isnull(df['About']))]
df.to_csv('dataClean.csv')
about_list = df['About']
channel = df['Channel name']
#Check if english, stem and lemmatize the descriptions
wordset = set(words.words())
is_english = []
token_word_list = []
for i, about in enumerate(about_list):
 #   print channel[i]
    token_words = word_tokenize(about)
    
    #take out non alpha-numeric entries, 'http' and 'https'
    bools = [token_words[i].isalnum() for i in range(len(token_words))]
    token_words = np.array(token_words)[bools].tolist()
    bools = [token_words[i]=='http' or token_words[i]=='https'for i in range(len(token_words))]
    token_words = np.array(token_words)[np.logical_not(bools)].tolist()
    
    token_word_list.append(token_words)
    #get what fraction of the words is english
    eng = 0
    for word in token_words:
        eng += word.lower() in wordset
#    print about  
    if len(token_words)==0:
        is_english.append(False)
#        print 'No words'
    elif np.float(eng)/np.float(len(token_words))<0.46:
        is_english.append(False)
#        print 'Not english: ' + str(np.float(eng)/np.float(len(token_words)))
    else:
        is_english.append(True)
#        print 'English: ' + str(np.float(eng)/np.float(len(token_words)))
#    
#    print '_______________________________________________________________'
    
df['IsEnglish'] = is_english
df['TokenWords'] = token_word_list
df.to_csv('dataTokensLang.csv')

df_eng = df[:][is_english]
token_word_list = df_eng['TokenWords']

#Stem and lemmatize the descriptions
porter = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
about_tokens = []
for i, token_words in enumerate(token_word_list):

    about_token = ''
    for word in token_words:
        stemmed_word = porter.stem(word)
        lemmed_word = wordnet_lemmatizer.lemmatize(stemmed_word)
        about_token = about_token+lemmed_word+' '
    about_tokens.append(about_token)
        
stop_words = ['an','any','are','as','at','and','be','by','for','has','have','if',
              'in','is','it','its','no','of','on','or','our','that','the',
              'their','this','to','us','was','we','where','which','with',
              'you','your']
vectorizer = CountVectorizer(max_features = 1000, stop_words=stop_words)
X = vectorizer.fit_transform(about_tokens)
wordlist = vectorizer.get_feature_names()
print wordlist
X=X.todense()

df_eng['TokenWordsCleaned'] = about_tokens
df_eng['Vector'] = X.tolist()
df_eng.to_csv('dataEng.csv')


np.savez('featureMatrix', X=X, token_word_list= token_word_list, 
         df_indeces = token_word_list.index, wordlist=wordlist)

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:12:13 2019

@author: swooty
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
import unicodedata
#import time
#import progressbar



df = pd.read_csv('data.csv')

#indL= 4000
#indH = 5000
channels = df['Channel name']#[indL:indH]
about_list = []


for ind, channel in enumerate(channels):
#for ind, channel in progressbar.progressbar(enumerate(channels)):
#    time.sleep(0.02)
    title = channel.replace(' ','').replace('-','')
    prestring = "https://www.youtube.com/user/"
    poststring = "/about?disable_polymer=1"

    r = requests.get(prestring+title+poststring, verify=False)
    page = r.text
    soup=bs(page,'html.parser')
    try:
        about = soup.pre.string
        about = unicodedata.normalize('NFKD', about).encode('ascii','ignore')
        about_list.append(about)
    
    except AttributeError:
        #print "Didn't work for " + channel
        about_list.append('nan')
        
badInd = [about_list[i] == 'nan' for i in range(len(about_list))]
print np.sum(badInd)/np.float(len(about_list))
        
df['About'] = about_list #[indL:indH]
df_clean = df[:][np.logical_not(badInd)]
df_clean = df_clean[:][np.logical_not(pd.isnull(df_clean['About']))]
df_clean.to_csv('data2.csv')
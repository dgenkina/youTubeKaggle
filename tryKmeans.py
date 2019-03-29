# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:36:10 2019

@author: swooty
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import time
from scipy import sparse

featureFile = np.load('featureMatrixSmall.npz')
X = featureFile['X']
wordlist = featureFile['wordlist']

K = 8
t1 = time.clock()
kmeans = KMeans(n_clusters=K,n_init = 100).fit(X)
t2 = time.clock()

print 'Seconds elapsed ' + str(t2-t1)

centroids = kmeans.cluster_centers_
centroids[centroids<1.0e-16]=0.0
sA = sparse.csr_matrix(centroids)

centroid_df = pd.DataFrame(wordlist, columns=['Words'])
for i in range(K):
    sort = np.argsort(centroids[i])
    weights = centroids[i][sort][::-1]
    words = wordlist[sort][::-1]
    centroid_df['Words_'+str(i)] = words
    centroid_df['Weights_'+str(i)] = weights
centroid_df.to_csv('kmeans_centroids.csv')
    


print kmeans.labels_
print kmeans.n_iter_
print kmeans.inertia_

df = pd.read_csv('data2.csv')
#df['Label'] = kmeans.labels_
from os import listdir
import csv

import scipy.io as sio

import pandas as pd

from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn.preprocessing as pp

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import DataUtil

# import importlib
# importlib.reload(Util)



#%% Import and store the data in layers


data = Util.MCTSData('./Blake Files/baserollout_and_10MCs/')
X = data.getLayer(4, ['xg_1', 'xg_2', 'xg_3', 'xg_4', 'xg_5', 'xg_6'])
scaler = pp.StandardScaler()
X = pd.DataFrame(data=scaler.fit_transform(X), columns=X.columns, index=X.index)


#%% kmeans cluster
from sklearn.cluster import KMeans

n_init = 10 # Number of times to run clustering with random centroids
max_iter = 500   # Number of iterations per run
tol = 0.0001    # Convergence tolerance
random_state = 2022

cluster_range = range(2, 30)
inertia = pd.DataFrame(data=[], columns=['inertia'], index=cluster_range)
sil_score = pd.DataFrame(data = [], columns=['silouette_avg'], index=cluster_range)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol)
    kmeans.fit(X)
    inertia.loc[n_clusters] = kmeans.score(X)

    clusters = kmeans.predict(X)
    clusters = pd.DataFrame(data=clusters, columns=['cluster'])

    sil_score.loc[n_clusters] = Util.silhouette_analysis(X, clusters, n_clusters)

plt.title('Interia vs. number of clusters')
plt.plot(cluster_range, inertia)
plt.show()
plt.title('Silhouette Score vs number of clusters')
plt.plot(cluster_range, sil_score, '.-')
plt.show()



#%% Closer look at fixed k for kmeans
k = 9
kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, tol=tol)
kmeans.fit(X)

clusters = kmeans.predict(X)
clusters = pd.DataFrame(data=clusters, columns=['predCluster'])

clust_data = []
states = ['xg_1', 'xg_2']
cmap = plt.cm.get_cmap('hsv', k)
for ii in range(k):
    clust_data.append(pd.DataFrame(data=X.loc[clusters[clusters.predCluster==ii].index, states]))
    plt.scatter(clust_data[ii][states[0]], clust_data[ii][states[1]], cmap=cmap, s=5, \
                label='Cluster ' + str(ii))

plt.legend(loc='upper left')
plt.show()


#%% DBSCAN
from sklearn.cluster import DBSCAN

eps_list = np.arange(.1, 1.5, .1)
for eps in eps_list:
    dbscan = DBSCAN(eps=eps, min_samples=10)

    dbscan.fit(X)

    unique, counts = np.unique(dbscan.labels_, return_counts=True)
    labels = dict(zip(unique,counts))
    # print('Eps = ', eps, ' - ', labels)
    print('Eps = {0:2.2f} - {1}'.format(eps, labels))


#%% Plot DBSCAN clusters
eps = 1.2
dbscan = DBSCAN(eps=eps, min_samples=15)
dbscan.fit(X)
clust_labels = pd.Series(data=dbscan.labels_)
unique, counts = np.unique(dbscan.labels_, return_counts=True)

states = ['xg_4', 'xg_5']
clustID = unique
cmap = plt.cm.get_cmap('hsv', len(unique))
for ii in clustID:
    clust_data = pd.DataFrame(data=X.loc[clust_labels[clust_labels==ii].index, states])
    plt.scatter(clust_data[states[0]], clust_data[states[1]], cmap=cmap, s=5, \
                label='Cluster ' + str(ii))

plt.legend(loc='best')
plt.show()


#%% Plot core/noncore
core = X.loc[dbscan.core_sample_indices_]
noncore = X.loc[dbscan.labels_ == -1]

plt.title('Samples with DBSCAN (Core = blue, Noncore = red): Eps=' + str(eps))
plt.xlabel('xg_1')
plt.ylabel('xg_2')
plt.plot(core['xg_1'], core['xg_2'], '.')
plt.plot(noncore['xg_1'], noncore['xg_2'], 'r.')
plt.show()
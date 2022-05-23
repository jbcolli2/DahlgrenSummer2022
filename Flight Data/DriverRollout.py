

from os import listdir
import csv

import scipy.io as sio

import pandas as pd

from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn.preprocessing as pp

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import Util

import importlib
importlib.reload(Util)



#%% Import and store the data in layers
file_names = './Blake Files/baserollout_and_10MCs/rollout_run.csv'
rollout = Util.Rollout(file_names)

layer = 5
states = ['xg_1', 'xg_2', 'xg_3', 'xg_4', 'xg_5', 'xg_6']
Xall = rollout.getLayers(layer)
X = rollout.getLayers(layer, states)
scaler = pp.StandardScaler()
X = pd.DataFrame(data=scaler.fit_transform(X), columns=X.columns, index=X.index)
# X.reset_index(inplace=True)

#%% kmeans cluster
from sklearn.cluster import KMeans

n_init = 30 # Number of times to run clustering with random centroids
max_iter = 500   # Number of iterations per run
tol = 0.0001    # Convergence tolerance


cluster_range = range(2, 14)
inertia = pd.DataFrame(data=[], columns=['inertia'], index=cluster_range)
sil_score = pd.DataFrame(data = [], columns=['silouette_avg'], index=cluster_range)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol)
    kmeans.fit(X)
    inertia.loc[n_clusters] = kmeans.score(X)

    clusters = kmeans.predict(X)
    clusters = pd.DataFrame(data=clusters, columns=['cluster'])

    sil_score.loc[n_clusters] = Util.silhouette_analysis(X, clusters, n_clusters, True)

plt.title('Interia vs. number of clusters | layer {}'.format(layer))
plt.plot(cluster_range, inertia)
plt.show()
plt.title('Silhouette Score vs number of clusters | layer {}'.format(layer))
plt.plot(cluster_range, sil_score, '.-')
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
eps = .9
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


#%% Gaussian Mixture
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=2, n_init=10)
gm.fit(X)

print('converge = ', gm.converged_)
print('num iters = ', gm.n_iter_)
unique, counts = np.unique(gm.predict(X), return_counts=True)
gm_labels = dict(zip(unique,counts))
print(gm_labels)

densities = gm.score_samples(X)
outliers = pd.DataFrame(data=[], columns=['num_outliers'])
for percent in range(4,100):
    density_threshold = np.percentile(densities, percent)
    outliers.loc[percent] = len(X[densities < density_threshold])


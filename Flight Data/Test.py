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
import Clustering as cl





#%%
dataDir = './data/Blake Files/baserollout_and_10MCs/'

data =DataUtil.MCTSData(dataDir)

# %%  KMeans clustering
layer = data.getRolloutLayers(2, DataUtil.xg_cols + DataUtil.x_cols)
# layer = data.getMCTSLayers(10, 2, DataUtil.xg_cols + DataUtil.x_cols)
layer.scaleData()

kMeans = cl.kMeans()
kMeans.displayMetrics(layer.X, list(range(2,11)), layer.getDescription())

kMeans.askHyperParameters()
kMeans.clusterData(layer.X, layer.getDescription())


kMeans.plotClusters(description=layer.getDescription())
kMeans.plotValueHist(layer, omega=1e5)










# %% DBSCAN clustering
layer = data.getRolloutLayers(2, DataUtil.xg_cols)


# layer = data.getMCTSLayers(6, 4, DataUtil.xg_cols + DataUtil.x_cols)
layer.scaleData()

dbscan = cl.dbscan()
dbscan.displayMetrics(layer.X, layer.getDescription())

dbscan.askHyperParameters()
dbscan.clusterData(layer.X, layer.getDescription())


dbscan.plotClusters(description=layer.getDescription())

layer.unscaleData()

layer.computeNodeValues(1e5)

clustVals = dict()
for key, clust in dbscan.clusters.items():
    clustVals[key] = layer.values.loc[clust.index,'nodeValue']
    print('\n-------------------Cluster {}: Size {}--------------------------'.format(key, len(clust)))
    print('Mean = {}'.format(np.mean(clustVals[key])))
    print('Median = {}'.format(np.median(clustVals[key])))
    print('Std = {}'.format(np.std(clustVals[key])))


# for values in clustVals:
#     v = values.values.flatten()
#     plt.hist(v, range=(0,1), bins=20, density=True)
#     print('Mean of ')

plt.show()










# %% Mean Shift
layer = data.getMCTSLayers(3,3, DataUtil.xg_cols)
layer.scaleData()

ms = cl.meanShift(quantile=0.3)
ms.displayMetrics(layer.X, layer.getDescription())
ms.askHyperParameters()
ms.clusterData(layer.X)
ms.plotClusters(description=layer.getDescription())

layer.computeNodeValues(1e5)

val = layer.values['value'].values.flatten()
valf = layer.values['termValue'].values.flatten()
plt.plot(val)
plt.plot(valf,'r')
plt.show()



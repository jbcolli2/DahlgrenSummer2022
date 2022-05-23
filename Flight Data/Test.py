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
import Clustering as cl





#%%
dataDir = './data/Blake Files/baserollout_and_10MCs/'

data = Util.MCTSData(dataDir)

# %%  KMeans clustering
# layer = data.getRolloutLayers(5, Util.xg_cols)
# layer = data.getMCTSLayers(10, 2, Util.xg_cols + Util.x_cols)
# layer.scaleData()
#
# kMeans = cl.kMeans()
# kMeans.displayMetrics(layer.X, list(range(2,11)), layer.getDescription())
#
# kMeans.askHyperParameters()
# kMeans.clusterData(layer.X, layer.getDescription())
#
#
# kMeans.plotClusters(description=layer.getDescription())

# %% DBSCAN clustering
layer = data.getRolloutLayers(5, Util.xg_cols)
layer = data.getRolloutLayers(5, Util.xg_cols + Util.x_cols)
layer = data.getMCTSLayers(3,3, Util.xg_cols)

# layer = data.getMCTSLayers(5, 3, Util.xg_cols + Util.x_cols)
layer.scaleData()

dbscan = cl.dbscan()
dbscan.displayMetrics(layer.X, layer.getDescription())

dbscan.askHyperParameters()
dbscan.clusterData(layer.X, layer.getDescription())


dbscan.plotClusters(description=layer.getDescription())


# %% Mean Shift
layer = data.getMCTSLayers(3,3, Util.xg_cols)
layer.scaleData()

ms = cl.meanShift(quantile=0.3)
ms.displayMetrics(layer.X, layer.getDescription())
ms.askHyperParameters()
ms.clusterData(layer.X)
ms.plotClusters(description=layer.getDescription())




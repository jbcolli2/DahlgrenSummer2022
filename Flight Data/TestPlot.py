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

import thinkplot as tp
import thinkstats2 as ts2





#%%
dataDir = './data/Blake Files/baserollout_and_10MCs/'

data = DataUtil.MCTSData(dataDir)

# %%  KMeans clustering
layer = data.getRolloutLayers(2, DataUtil.xg_cols + DataUtil.x_cols)
layer.scaleData()

kMeans = cl.kMeans()
cl.runCluster(kMeans, layer, list(range(2,11)) )


# %% Plot the value histogram

layer.unscaleData()
kMeans.clusterData(layer.X, layer.getDescription())

index = 0
clusterVals = []
for cluster in kMeans.clusters:
    clusterVals.append(DataUtil.computeNodeValues(kMeans.clusters[index], layer.Xall, 1e5))
    index += 1

numRows, numCols = 2,2
nBins = 20


fig, ax = plt.subplots(numRows, numCols, figsize=(30,30))
fig.set_size_inches(18, 18)

clusterIdx = 0
for row in range(numRows):
    for col in range(numCols):
        clusterVals[clusterIdx].nodeValue.plot.hist(density=True, bins=nBins, ax=ax[row, col])
        clusterVals[clusterIdx].nodeValue.plot.kde(ax=ax[row, col], title='Cluster {} | Size = {}'
                                                   .format(clusterIdx, len(clusterVals[clusterIdx])))
        ax[row,col].set_xlim([0,1])
        clusterIdx += 1

plt.suptitle(layer.getDescription())
fig.show()


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
rollLayer = data.getRolloutLayers(2, DataUtil.xg_cols + DataUtil.x_cols)
treeLayer = data.getMCTSLayers(9, 5, DataUtil.xg_cols + DataUtil.x_cols)

gm = cl.gaussian(n_init=30)
nCluster_range = list(range(2, 20))
gm.runClustering(rollLayer, nCluster_range)


# Plot results
print(gm.computeClusterStats(1e5, val_bounds=[0, 1]))

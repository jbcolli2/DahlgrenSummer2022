from os import listdir
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

import sklearn.preprocessing as pp

from enum import Enum

xg_cols = ['xg_1', 'xg_2', 'xg_3', 'xg_4', 'xg_5', 'xg_6']
x_cols = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6']
u_cols = ['u0_1', 'u0_2']
c_cols = ['c_1', 'c_2', 'c_3', 'c_4', 'c_5']
cf_cols = ['cf_1', 'cf_2', 'cf_3', 'cf_4', 'cf_5', 'cf_6']

'''
    Holds all data within a directory.  Has access methods to obtain either a set of layers or paths
        from one particular file, either a rollout or iteration file.
'''
class MCTSData:

    # Ctor passed the directory of the data and loads all the data into members
    def __init__(self, dir_str, iteration_names=None, rollout_name=None):
        self.dir_str = dir_str

        if(iteration_names == None):
            file_names = listdir(dir_str)
            self.iteration_names = [s for s in file_names if s[0] == 'M']
            self.iteration_names = [dir_str + s for s in self.iteration_names]
            self.rollout_name = dir_str + file_names[-1]
        else:
            self.iteration_names = iteration_names
            self.rollout_name = rollout_name


        self.mcts_data = [None] #mcts_data is a list holding each iterations data
        for ii in range(0, len(self.iteration_names)):
            self.mcts_data.append(MCTSTree(self.iteration_names[ii]))

        self.rollout_data = Rollout(self.rollout_name)




    def getRolloutLayers(self, layers: int, cols = slice(None)):

        return self.rollout_data.getLayers(layers, cols)

    def getRolloutPaths(self, iteration, paths, cols):
        return self.mcts_data[iteration].getPaths(paths, cols)


    '''
        Access a data of an iteration tree
    '''
    # Returns every node in layerID with all the columns
    def getMCTSLayers(self, iteration: int, layers: int, cols = slice(None)):
        return self.mcts_data[iteration].getLayers(layers, cols)

    def getMCTSPaths(self, iteration, paths, cols):
        return self.mcts_data[iteration].getPaths(paths, cols)






'''
    Holds data for one iteration of the MCTS run.  Holds the entire tree
'''
class MCTSTree:
    def __init__(self, mcts_filename):
        self.filename = mcts_filename
        self.original_data = pd.read_csv(mcts_filename)
        self.original_data = self.original_data.dropna(how='all', subset=['layer'])

        self.data = self.original_data.drop(columns=['layer', 'id', 'parent'])
        ind = pd.MultiIndex.from_tuples([], names=['path', 'layer'])
        self.paths = pd.DataFrame(data=[], columns=self.data.columns, index=ind)

        roots = self.original_data.loc[self.original_data.parent == 0]
        if(len(roots) == 1):  # Only one root means it is a US tree structure
            self.computePaths()
        else:  # We have the tree in Japanese format
            japMultiIdx = pd.MultiIndex.from_arrays([self.original_data.id.values, self.original_data.layer.values],
                                                    names=['path','layer'])
            self.paths = pd.DataFrame(data=self.original_data.values, columns=self.original_data.columns,
                                      index=japMultiIdx)
            self.paths = self.paths.drop(columns=['layer','id','parent'])






    '''
        Access data
    '''

    # Return all nodes from a particular layer or set of layers
    def getLayers(self, layers, cols = slice(None)):
        idx = pd.IndexSlice
        return Data(self.paths.loc[idx[:, layers], cols], self.paths, self.filename, DataType.LAYER, layers, cols)
        # return self.data.loc[self.original_data.layer == layer, cols]

    # Return a collection of full paths with certain columns
    def getPaths(self, paths, cols = slice(None)):
        return Data(self.paths.loc[paths, cols], self.paths, self.filename, DataType.PATH, paths, cols)

    # Return all the data, from certain columns
    def getData(self, cols = slice(None)):
        return Data(self.paths[cols], self.paths, self.filename, DataType.ALL, None, cols)


    # Get paths from the data
    def computePaths(self):
        root = self.original_data.loc[0,:]
        path = pd.DataFrame(data=[], columns=self.data.columns, index=self.paths.index)
        self.pathCounter = 0
        self.getChild(path, root)


    def getChild(self, path, root):
        path.loc[(self.pathCounter, root.layer),:] = root

        child_idx = self.original_data[(self.original_data.parent == root.id) & (self.original_data.layer == (root.layer+1))].index

        for idx in child_idx:
             self.getChild(path, self.original_data.loc[idx,:])


        if(len(child_idx) == 0):
            self.paths = self.paths.append(path)
            self.pathCounter += 1
            if(self.pathCounter % 100 == 0):
                print('Computed {} paths!'.format(self.pathCounter))
            path.index = path.index.set_levels([self.pathCounter], level=0)

        path = path.drop(path.iloc[-1,:].name)





''' 
    Holds all the rollout data in a multiIndex dataframe organized by paths.
'''

class Rollout:
    def __init__(self, rollout_filename):
        self.filename = rollout_filename
        self.original_data = pd.read_csv(rollout_filename)
        self.original_data = self.original_data.dropna(how='all')
        self.data = self.original_data.reset_index()

        # Refomat into paths
        multiIdx = pd.MultiIndex.from_product([np.arange(1, int(self.data.shape[0] / 5) + 1, 1), [1, 2, 3, 4, 5]],
                                              names=['path', 'layer'])
        self.data = pd.DataFrame(data=self.data.values, index=multiIdx, columns=self.data.columns)
        self.dataIndex = pd.Series(data=self.data['index'], index=multiIdx)

        self.data = self.data.drop(columns=['index'])

    '''
        Access the data
    '''

    # Get a bunch of paths with all layers
    def getPaths(self, paths, cols=slice(None)):
        idx = pd.IndexSlice
        return Data(self.data.loc[idx[paths, cols]], self.data, self.filename, DataType.PATH, paths, cols)

    # Get all nodes from some layers
    def getLayers(self, layers, cols=slice(None)):
        idx = pd.IndexSlice
        return Data(self.data.loc[idx[:, layers], cols], self.data, self.filename, DataType.LAYER, layers, cols)

    # Get full data set
    def getData(self, cols=slice(None)):
        return Data(self.data[cols], self.data, self.filename, DataType.ALL, None, cols)


'''
    Proxy class to hold one dataframe to be used on clustering.  This DataFrame could be 
        a set of layers from one of the iterations or rollouts, or it could be many layers
        or paths from one of those files.
        
    Purpose of this proxy class is to also hold metadata for the DataFrame.  Things like the 
        filename where the data came from, if it is layer or path data, and if so what is the constant
        value (layerID or pathID, or set of those).  This is to be used in plotting.  Main data is always
        held in X variable
'''

class DataType(Enum):
    LAYER = "Layer"
    PATH  = "Path"
    ALL = "All"
    OTHER = "Other"



class Data:
    def __init__(self, X, Xall, filename, type = None, typeValue = None, cols = None):
        self.X = X
        self.Xall = Xall
        self.filename = filename
        if(type == None):
            self.type = DataType.OTHER
        else:
            self.type = type

        self.typeValue = typeValue
        self.cols = cols



    def scaleData(self):
        self.scaler = pp.StandardScaler()
        self.X = pd.DataFrame(data=self.scaler.fit_transform(self.X), columns=self.X.columns, index=self.X.index)

    def unscaleData(self):
        self.X = pd.DataFrame(data=self.scaler.inverse_transform(self.X), columns=self.X.columns, index=self.X.index)

    def getDescription(self):
        sType = self.type.name
        sValue = ""
        try:
            sValue = str(self.typeValue[0]) + "..." + str(self.typeValue[-1])
        except TypeError:
            sValue = str(self.typeValue)
        sCols = ""
        try:
            sCols = str(self.cols[0]) + ', ' + str(self.cols[1]) + ',...,' + str(self.cols[-2]) + ', ' + str(self.cols[-1])
        except TypeError:
            sCols = str(self.cols)

        return self.filename + '  |  ' + str(sType) + ' ' + sValue +'\n' + sCols




'''
    Analyze a cluster by computing it's average silhouette value and the 
    silhouette value of each of the instances in all the clusters
    
    If `plot_sil_diagram=True` then it will plot a silhouette diagram.  This can 
        become unwieldy for a large number of clusters.
'''
def silhouette_analysis(X, cluster_labels, n_clusters, plot_sil_diagram=False):
    cluster_labels = cluster_labels.values.flatten()




    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    if(not plot_sil_diagram):
        return silhouette_avg


    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(7, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])


    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)




    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    return silhouette_avg
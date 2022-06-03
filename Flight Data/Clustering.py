import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

import DataUtil


class hyperParams:
    def __init__(self):
        self.description = 'with abstract hyperparameters'



class hpNumClusters(hyperParams):
    def __init__(self, n_clusters = 2):
        super().__init__()
        self.description = 'NumClusters = {}'.format(n_clusters)
        self.n_clusters = n_clusters


    @property
    def n_clusters(self):
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, n_clusters):
        self._n_clusters = n_clusters
        self.description = ' with NumClusters = {}'.format(n_clusters)

class hpDBSCAN(hyperParams):
    def __init__(self, eps = .5):
        super().__init__()
        self.eps = eps

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, eps):
        self._eps = eps
        self.description = ' with Epsilon = {}'.format(eps)

class hpMeanShift(hyperParams):
    def __init__(self, bandwidth):
        super().__init__()
        self.bandwidth = None

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        self._bandwidth = bandwidth
        self.description = ' with NumClusters = {}'.format(bandwidth)







class clustAlg:
    def __init__(self, name):
        self.name = name
        self.clusters = dict()
        self.hyper = hyperParams()
        self.dataObj = None

    def displayMetrics(self, X, hp_range, description):
        print('Abstract displayMetrics')
        pass

    def askHyperParameters(self):
        print('Abstract askHyper')
        pass

    def clusterData(self, X, description):
        print('Abstract clusterData')
        pass



    def runClustering(self, dataObj, hp_range=None):
        dataObj.scaleData()
        self.displayMetrics(dataObj.X, hp_range, dataObj.getDescription())
        self.askHyperParameters()
        self.clusterData(dataObj)
        dataObj.unscaleData()


    # Plot the clusters in 2D scatter plot
    def plotClusters(self, cols=None):
        if cols == None:
            cols = self.clusters[0].columns

        if len(cols) > 6:
            cols = cols[:6]

        plotDescription = self.name + self.hyper.description + '\n' + self.dataObj.getDescription()

        numPlotCols = int(np.ceil(np.sqrt(len(cols))))
        fig, ax = plt.subplots(len(cols), len(cols), figsize=(30, 30))
        fig.set_size_inches(18, 18)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        cmap = plt.cm.get_cmap('hsv', len(self.clusters))
        for plotRow in range(len(cols)):
            for plotCol in range(len(cols)):
                if (plotRow != plotCol):
                    for key, cluster in self.clusters.items():
                        ax[plotRow, plotCol].scatter(cluster[cols[plotRow]], cluster[cols[plotCol]],
                                                     cmap=cmap, s=5, label='C ' + str(key))
                        ax[plotRow, plotCol].set_xlabel(cols[plotRow])
                        ax[plotRow, plotCol].set_ylabel(cols[plotCol])
                        ax[plotRow, plotCol].ticklabel_format(scilimits=(0, 0))

        handles, labels = ax.flatten()[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left', fontsize=20)
        plt.suptitle(plotDescription, fontsize=20)
        fig.show()


    # Plot a histogram of the values of each of the clusters
    def plotValueHist(self, omega, nBins = 20):
        bins = np.arange(0, 1 + 1/nBins, 1/nBins)

        self.dataObj.unscaleData()

        clusterVals = dict()
        for key, cluster in self.clusters.items():
            clusterVals[key] = DataUtil.computeNodeValues(cluster, self.dataObj.Xall, omega)

        numRows = int(np.ceil(np.sqrt(len(clusterVals))))
        numCols = numRows

        fig, ax = plt.subplots(numRows, numCols, figsize=(30, 30))
        fig.set_size_inches(18, 18)

        clusterIter = iter(clusterVals)
        clusterIdx = next(clusterIter)
        for row in range(numRows):
            for col in range(numCols):
                clusterVals[clusterIdx].nodeValue.plot.hist(density=True, bins=bins, ax=ax[row, col])
                clusterVals[clusterIdx].nodeValue.plot.kde(ax=ax[row, col], title='Cluster {} | Size = {}'
                                                           .format(clusterIdx, len(clusterVals[clusterIdx])))
                ax[row, col].set_xlim([0, 1])
                try:
                    clusterIdx = next(clusterIter)
                except StopIteration:
                    break

        plt.suptitle(self.name + self.hyper.description + '\n' + self.dataObj.getDescription() + '\n Omega = ' + str(omega))
        fig.show()



    # Compute the stats such as mean, median and standard deviation of the value for each cluster and store in DataFrame
    def computeClusterStats(self, omega, val_bounds=[0,1], std_bounds=[0,.3]):
        clusterValStats = pd.DataFrame(data=[], columns=['mean', 'median', 'std_dev'], index=self.clusters.keys())


        clusterVals = dict()
        for key, cluster in self.clusters.items():
            clusterVals[key] = DataUtil.computeNodeValues(cluster, self.dataObj.Xall, omega)
            clusterValStats.loc[key, 'mean'] = np.mean(clusterVals[key].nodeValue)
            clusterValStats.loc[key, 'median'] = np.median(clusterVals[key].nodeValue)
            clusterValStats.loc[key, 'std_dev'] = np.std(clusterVals[key].nodeValue)

        median = clusterValStats['median'].values
        median = np.sort(median)
        mean = clusterValStats['mean'].values
        mean = np.sort(mean)
        std = clusterValStats['std_dev'].values
        std = np.sort(std)

        fig, ax = plt.subplots(2, 2, figsize=(30, 30))
        fig.set_size_inches(18, 18)

        ax[0,0].plot(median, '.-')
        ax[0,0].set_title('Median of {} Clusters'.format(len(clusterVals)))
        ax[0,0].set_ylim(val_bounds)
        ax[0,1].plot(mean, 'r.-')
        ax[0,1].set_title('Mean of {} Clusters'.format(len(clusterVals)))
        ax[0,1].set_ylim(val_bounds)
        ax[1,0].plot(std, 'g.-')
        ax[1,0].set_title('StdDev of {} Clusters'.format(len(clusterVals)))
        ax[1,0].set_ylim(std_bounds)

        sortedMean = clusterValStats.sort_values(by=['median'])
        sortedMean.reset_index(inplace=True)
        boxwhis_data = []
        for clustIdx in sortedMean['index']:
            boxwhis_data.append(clusterVals[clustIdx]['nodeValue'].values)


        ax[1,1].boxplot(boxwhis_data, whis=(0,100))
        ax[1,1].set_title('Box and Whisker plot of node values | {} Clusters'.format(len(clusterVals)))
        ax[1,1].set_ylim(val_bounds)
        ax[1,1].set_xticks(range(len(boxwhis_data)+1))
        ax[1,1].set_xticklabels([' '] + list(sortedMean['index'].values))

        plt.suptitle(self.name + self.hyper.description + '\n' + self.dataObj.getDescription())
        fig.show()
        return clusterValStats




# %% KMEANS
class kMeans(clustAlg):
    def __init__(self, n_init = 10, max_iter = 300, tol = 1e-4):
        super().__init__('KMeans')

        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

        # Hyperparameters for kmeans
        self.hyper = hpNumClusters()


    # Run kmeans for many different cluster numbers and display the inertia and sil_score for each
    #   number of clusters.  This will be plotted to determine the correct number of clusters to use
    def displayMetrics(self, X, hp_range, description = "Unknown Data"):
        inertia = pd.DataFrame(data=[], columns=['inertia'], index=hp_range)
        sil_score = pd.DataFrame(data=[], columns=['silouette_avg'], index=hp_range)

        for n_clusters in hp_range:
            kmeans = KMeans(n_clusters=n_clusters, n_init=self.n_init, max_iter=self.max_iter, tol=self.tol)
            kmeans.fit(X)
            inertia.loc[n_clusters] = kmeans.inertia_

            clusters_predict = kmeans.predict(X)
            clusters_predict = pd.DataFrame(data=clusters_predict, columns=['cluster'], index=X.index)

            sil_score.loc[n_clusters] = DataUtil.silhouette_analysis(X, clusters_predict, n_clusters)

            print('For {} clusters: Avg Sill = {} and Inertia = {}'.format(n_clusters, sil_score.loc[n_clusters,'silouette_avg'],
                                                                           inertia.loc[n_clusters,'inertia']))

        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(18, 12)
        plt.suptitle(description)
        ax1.set_title('Interia vs. number of clusters')
        ax1.plot(hp_range, inertia)
        ax2.set_title('Silhouette Score vs number of clusters')
        ax2.grid(axis='x')
        ax2.plot(hp_range, sil_score, '.-')
        fig.show()


    # Obtain the number of hyperparameters to use from the user
    def setHyperParameters(self, n_clusters: int):
        self.hyper.n_clusters = n_clusters

    def askHyperParameters(self):
        self.hyper.n_clusters = int(input("Enter number of clusters for KMeans: "))


    # Actually cluster the data and store the clusters in a list of DataFrames called `clusters`
    def clusterData(self, dataObj: DataUtil.Data, description="Unknown Data"):
        self.dataObj = dataObj
        X = self.dataObj.X

        kmeans = KMeans(n_clusters=self.hyper.n_clusters, n_init=self.n_init, max_iter=self.max_iter, tol=self.tol)
        kmeans.fit(X)
        self.predClusterLabels = pd.DataFrame(kmeans.predict(X), index=X.index, columns=['cluster'])

        # Create a list of DataFrames containing all the clusters of the data
        self.clusters = dict()
        for clusterIdx in range(self.hyper.n_clusters):
            self.clusters[clusterIdx] = X.loc[self.predClusterLabels.cluster == clusterIdx]







from sklearn.neighbors import NearestNeighbors
class dbscan(clustAlg):
    def __init__(self, min_samples = 5):
        super().__init__('DBSCAN')


        self.min_samples = min_samples

        # Hyperparameters for kmeans
        self.hyper = hpDBSCAN()


    # Run kmeans for many different cluster numbers and display the inertia and sil_score for each
    #   number of clusters.  This will be plotted to determine the correct number of clusters to use


    def displayMetrics(self, X, hp_range, description = "Unknown Data"):

        '''
            Create plot of distance to nearest neighbors to determine epislon.  From this website
            https://medium.com/towards-data-science/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
        '''
        NNeighAlg = NearestNeighbors(n_neighbors=2)
        nbrs = NNeighAlg.fit(X)
        distances, indices = nbrs.kneighbors(X)

        distances = np.sort(distances, axis=0)
        plt.plot(list(range(len(distances))), distances[:,1])
        plt.title('DBSCAN Nearest Neighbor Plot\n' + description)
        plt.grid(axis='y')
        plt.show()







    # Obtain the number of hyperparameters to use from the user
    def setHyperParameters(self, eps: float):
        self.hyper.eps = eps

    def askHyperParameters(self):
        self.hyper.eps = float(input("Enter epsilon value for DBSCAN: "))


    # Actually cluster the data and store the clusters in a list of DataFrames called `clusters`
    def clusterData(self, dataObj: DataUtil.Data, description="Unknown Data"):
        self.dataObj = dataObj
        X = self.dataObj.X

        dbscan = DBSCAN(eps=self.hyper.eps, min_samples=self.min_samples)
        dbscan.fit(X)

        unique, counts = np.unique(dbscan.labels_, return_counts=True)
        self.clusterCount = dict(zip(unique, counts))
        print('Cluster counts for epsilon = {}: {}'.format(self.hyper.eps, self.clusterCount))


        self.predClusterLabels = pd.DataFrame(dbscan.labels_, index=X.index, columns=['cluster'])

        # Create a list of DataFrames containing all the clusters of the data
        self.clusters = dict()
        for clusterIdx in unique:
            self.clusters[clusterIdx] = X.loc[self.predClusterLabels.cluster == clusterIdx]




    #     fig.show()


from sklearn.mixture import GaussianMixture
class gaussian(clustAlg):
    def __init__(self, n_init = 10, max_iter = 300, tol = 1e-4):
        super().__init__('Gaussian Mixture')

        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

        # Hyperparameters for kmeans
        self.hyper = hpNumClusters()


    # Run kmeans for many different cluster numbers and display the inertia and sil_score for each
    #   number of clusters.  This will be plotted to determine the correct number of clusters to use
    def displayMetrics(self, X, hp_range, description = "Unknown Data"):
        bic = pd.DataFrame(data=[], columns=['BIC'], index=hp_range)
        aic = pd.DataFrame(data=[], columns=['AIC'], index=hp_range)

        for n_clusters in hp_range:
            gauss = GaussianMixture(n_components=n_clusters, n_init=self.n_init, max_iter=self.max_iter, tol=self.tol)
            gauss.fit(X)
            if not gauss.converged_:
                print('Gaussian Mixture did not converge with {} clusters'.format(n_clusters))

            bic.loc[n_clusters] = gauss.bic(X)
            aic.loc[n_clusters] = gauss.aic(X)



        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(18, 12)
        plt.suptitle(description)
        ax1.set_title('BIC vs. number of clusters')
        ax1.plot(cluster_range, bic)
        ax2.set_title('AIC vs number of clusters')
        ax2.plot(cluster_range, aic, '.-')
        fig.show()


    # Obtain the number of hyperparameters to use from the user
    def setHyperParameters(self, n_clusters: int):
        self.hyper.n_clusters = n_clusters

    def askHyperParameters(self):
        self.hyper.n_clusters = int(input("Enter number of clusters for Gaussian Mixture: "))


    # Actually cluster the data and store the clusters in a list of DataFrames called `clusters`
    def clusterData(self, dataObj: DataUtil.Data, description="Unknown Data"):
        self.dataObj = dataObj
        X = self.dataObj.X

        self.gauss = GaussianMixture(n_components=self.hyper.n_clusters, n_init=self.n_init, max_iter=self.max_iter, tol=self.tol)
        self.gauss.fit(X)
        predict = self.gauss.predict_proba(X)

        # Create a list of DataFrames containing all the clusters of the data
        self.clusters = dict()
        outliers = X
        for clusterIdx in range(self.hyper.n_clusters):
            cluster = X.loc[predict[:,clusterIdx] > .66]
            self.clusters[clusterIdx] = cluster
            outliers = outliers.drop(index=cluster.index)

        self.clusters[-1] = outliers





    # Plot the clusters in 2D scatter plot
    # def plotClusters(self, cols = None, description=None):
    #     if cols == None:
    #         cols = self.clusters[0].columns
    #
    #     if len(cols) > 6:
    #         cols = cols[:6]
    #
    #     plotDescription = 'Gaussian Mixture with k = ' + str(self.hyper.n_clusters) + '\n' + description
    #
    #     numPlotCols = int(np.ceil(np.sqrt(len(cols))))
    #     fig, ax = plt.subplots(len(cols), len(cols), figsize=(30,30))
    #     fig.set_size_inches(18, 18)
    #     plt.subplots_adjust(wspace=0.5, hspace=0.5)
    #     cmap = plt.cm.get_cmap('hsv', len(self.clusters))
    #     for plotRow in range(len(cols)):
    #         for plotCol in range(len(cols)):
    #             if(plotRow != plotCol):
    #                 for key, cluster in self.clusters.items():
    #                     ax[plotRow, plotCol].scatter(cluster[cols[plotRow]], cluster[cols[plotCol]],
    #                                                  cmap=cmap, s=5, label='C ' + str(key))
    #                     ax[plotRow, plotCol].set_xlabel(cols[plotRow])
    #                     ax[plotRow, plotCol].set_ylabel(cols[plotCol])
    #                     ax[plotRow, plotCol].ticklabel_format(scilimits=(0,0))
    #
    #     handles, labels = ax.flatten()[1].get_legend_handles_labels()
    #     fig.legend(handles, labels, loc='upper left')
    #     plt.suptitle(plotDescription)
    #     fig.show()

from sklearn.cluster import MeanShift, estimate_bandwidth
class meanShift(clustAlg):
    def __init__(self, quantile=0.5):
        super().__init__('MeanShift')

        self.batch_n = None
        self.quantile = quantile

        # Hyperparameters for kmeans
        self.hyper = hpMeanShift(None)

    # Run kmeans for many different cluster numbers and display the inertia and sil_score for each
    #   number of clusters.  This will be plotted to determine the correct number of clusters to use
    def displayMetrics(self, X, hp_range, description="Unknown Data"):
        return

    # Obtain the number of hyperparameters to use from the user
    def setHyperParameters(self, bandwidth: float):
        self.hyper.bandwidth = bandwidth

    def askHyperParameters(self):
        return

    # Actually cluster the data and store the clusters in a list of DataFrames called `clusters`
    def clusterData(self, dataObj: DataUtil.Data, description="Unknown Data"):
        self.dataObj = dataObj
        X = self.dataObj.X

        self.hyper.bandwidth = estimate_bandwidth(X, quantile=self.quantile, n_samples=self.batch_n)
        self.ms = MeanShift(bandwidth=self.hyper.bandwidth)
        self.ms.fit(X)

        unique, counts = np.unique(self.ms.labels_, return_counts=True)
        self.clusterCount = dict(zip(unique, counts))
        print('Cluster counts for bandwidth = {}: {}'.format(self.hyper.bandwidth, self.clusterCount))

        self.predClusterLabels = pd.DataFrame(self.ms.labels_, index=X.index, columns=['cluster'])

        # Create a list of DataFrames containing all the clusters of the data
        self.clusters = dict()
        for clusterIdx in unique:
            self.clusters[clusterIdx] = X.loc[self.predClusterLabels.cluster == clusterIdx]







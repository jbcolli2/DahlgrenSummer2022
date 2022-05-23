import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

import Util

class hpKMeans:
    def __init__(self, n_clusters = 2):
        self.n_clusters = n_clusters

class hpDBSCAN:
    def __init__(self, eps = .5):
        self.eps = eps



class kMeans:
    def __init__(self, n_init = 10, max_iter = 300, tol = 1e-4):
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

        # Hyperparameters for kmeans
        self.hyper = hpKMeans()


    # Run kmeans for many different cluster numbers and display the inertia and sil_score for each
    #   number of clusters.  This will be plotted to determine the correct number of clusters to use
    def displayMetrics(self, X, cluster_range, description = "Unknown Data"):
        inertia = pd.DataFrame(data=[], columns=['inertia'], index=cluster_range)
        sil_score = pd.DataFrame(data=[], columns=['silouette_avg'], index=cluster_range)

        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, n_init=self.n_init, max_iter=self.max_iter, tol=self.tol)
            kmeans.fit(X)
            inertia.loc[n_clusters] = kmeans.inertia_

            clusters_predict = kmeans.predict(X)
            clusters_predict = pd.DataFrame(data=clusters_predict, columns=['cluster'], index=X.index)

            sil_score.loc[n_clusters] = Util.silhouette_analysis(X, clusters_predict, n_clusters)

        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(18, 12)
        plt.suptitle(description)
        ax1.set_title('Interia vs. number of clusters')
        ax1.plot(cluster_range, inertia)
        ax2.set_title('Silhouette Score vs number of clusters')
        ax2.plot(cluster_range, sil_score, '.-')
        fig.show()


    # Obtain the number of hyperparameters to use from the user
    def setHyperParameters(self, n_clusters: int):
        self.hyper.n_clusters = n_clusters

    def askHyperParameters(self):
        self.hyper.n_clusters = int(input("Enter number of clusters for KMeans: "))


    # Actually cluster the data and store the clusters in a list of DataFrames called `clusters`
    def clusterData(self, X: pd.DataFrame, description="Unknown Data"):
        kmeans = KMeans(n_clusters=self.hyper.n_clusters, n_init=self.n_init, max_iter=self.max_iter, tol=self.tol)
        kmeans.fit(X)
        self.predClusterLabels = pd.DataFrame(kmeans.predict(X), index=X.index, columns=['cluster'])

        # Create a list of DataFrames containing all the clusters of the data
        self.clusters = []
        for clusterIdx in range(self.hyper.n_clusters):
            self.clusters.append(X.loc[self.predClusterLabels.cluster == clusterIdx])




    # Plot the clusters in 2D scatter plot
    def plotClusters(self, cols = None, description=None):
        if cols == None:
            cols = self.clusters[0].columns

        if len(cols) > 6:
            cols = cols[:6]

        plotDescription = 'KMeans with k = ' + str(self.hyper.n_clusters) + '\n' + description

        numPlotCols = int(np.ceil(np.sqrt(len(cols))))
        fig, ax = plt.subplots(len(cols), len(cols), figsize=(30,30))
        fig.set_size_inches(18, 18)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        cmap = plt.cm.get_cmap('hsv', len(self.clusters))
        for plotRow in range(len(cols)):
            for plotCol in range(len(cols)):
                if(plotRow != plotCol):
                    for cluster in range(len(self.clusters)):
                        ax[plotRow, plotCol].scatter(self.clusters[cluster][cols[plotRow]], self.clusters[cluster][cols[plotCol]],
                                                     cmap=cmap, s=5)
                        ax[plotRow, plotCol].set_xlabel(cols[plotRow])
                        ax[plotRow, plotCol].set_ylabel(cols[plotCol])
                        ax[plotRow, plotCol].ticklabel_format(scilimits=(0,0))

        plt.suptitle(description)
        fig.show()


from sklearn.neighbors import NearestNeighbors
class dbscan:
    def __init__(self, min_samples = 5):
        self.min_samples = min_samples

        # Hyperparameters for kmeans
        self.hyper = hpDBSCAN()


    # Run kmeans for many different cluster numbers and display the inertia and sil_score for each
    #   number of clusters.  This will be plotted to determine the correct number of clusters to use


    def displayMetrics(self, X, description = "Unknown Data"):

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
        plt.show()




        # fig, (ax1, ax2) = plt.subplots(1,2)
        # fig.set_size_inches(18, 12)
        # plt.suptitle(description)
        # ax1.set_title('Interia vs. number of clusters')
        # ax1.plot(cluster_range, inertia)
        # ax2.set_title('Silhouette Score vs number of clusters')
        # ax2.plot(cluster_range, sil_score, '.-')
        # fig.show()


    # Obtain the number of hyperparameters to use from the user
    def setHyperParameters(self, eps: float):
        self.hyper.eps = eps

    def askHyperParameters(self):
        self.hyper.eps = float(input("Enter epsilon value for DBSCAN: "))


    # Actually cluster the data and store the clusters in a list of DataFrames called `clusters`
    def clusterData(self, X: pd.DataFrame, description="Unknown Data"):
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




    # Plot the clusters in 2D scatter plot
    def plotClusters(self, cols = None, description=None):
        if cols == None:
            cols = self.clusters[0].columns

        if len(cols) > 6:
            cols = cols[:6]

        plotDescription = 'DBSCAN with eps = ' + str(self.hyper.eps) + '\n' + description

        numPlotCols = int(np.ceil(np.sqrt(len(cols))))
        fig, ax = plt.subplots(len(cols), len(cols), figsize=(30,30))
        fig.set_size_inches(18, 18)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        cmap = plt.cm.get_cmap('hsv', len(self.clusters))
        for plotRow in range(len(cols)):
            for plotCol in range(len(cols)):
                if(plotRow != plotCol):
                    for key, cluster in self.clusters.items():
                        ax[plotRow, plotCol].scatter(cluster[cols[plotRow]], cluster[cols[plotCol]],
                                                     cmap=cmap, s=5, label='C ' + str(key))
                        ax[plotRow, plotCol].set_xlabel(cols[plotRow])
                        ax[plotRow, plotCol].set_ylabel(cols[plotCol])
                        ax[plotRow, plotCol].ticklabel_format(scilimits=(0,0))


        handles, labels = ax.flatten()[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')
        plt.suptitle(description)
        fig.show()


from sklearn.mixture import GaussianMixture
class gaussian:
    def __init__(self, n_init = 10, max_iter = 300, tol = 1e-4):
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

        # Hyperparameters for kmeans
        self.hyper = hpKMeans()


    # Run kmeans for many different cluster numbers and display the inertia and sil_score for each
    #   number of clusters.  This will be plotted to determine the correct number of clusters to use
    def displayMetrics(self, X, cluster_range, description = "Unknown Data"):
        bic = pd.DataFrame(data=[], columns=['BIC'], index=cluster_range)
        aic = pd.DataFrame(data=[], columns=['AIC'], index=cluster_range)

        for n_clusters in cluster_range:
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
    def clusterData(self, X: pd.DataFrame, description="Unknown Data"):
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
    def plotClusters(self, cols = None, description=None):
        if cols == None:
            cols = self.clusters[0].columns

        if len(cols) > 6:
            cols = cols[:6]

        plotDescription = 'Gaussian Mixture with k = ' + str(self.hyper.n_clusters) + '\n' + description

        numPlotCols = int(np.ceil(np.sqrt(len(cols))))
        fig, ax = plt.subplots(len(cols), len(cols), figsize=(30,30))
        fig.set_size_inches(18, 18)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        cmap = plt.cm.get_cmap('hsv', len(self.clusters))
        for plotRow in range(len(cols)):
            for plotCol in range(len(cols)):
                if(plotRow != plotCol):
                    for key, cluster in self.clusters.items():
                        ax[plotRow, plotCol].scatter(cluster[cols[plotRow]], cluster[cols[plotCol]],
                                                     cmap=cmap, s=5, label='C ' + str(key))
                        ax[plotRow, plotCol].set_xlabel(cols[plotRow])
                        ax[plotRow, plotCol].set_ylabel(cols[plotCol])
                        ax[plotRow, plotCol].ticklabel_format(scilimits=(0,0))

        handles, labels = ax.flatten()[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')
        plt.suptitle(description)
        fig.show()
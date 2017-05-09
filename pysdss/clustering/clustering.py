# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        clustering.py
# Purpose:     clustering field data
#
# Author:      claudio piccinini
#
# Updated:     23/02/2017
#-------------------------------------------------------------------------------


from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from pysdss.utility import vis


def kmeans_cluster(df, x, y, n_clusters, n_jobs=-1, viz=True, index=-4, allcolumns= True, group=True, minibatch = True):
    """
    Cluster and visualize a dataset,
    important: the columns must be on the right

    :param df: pandas dataframe
    :param x: array with coordinates
    :param y: array with coordinates
    :param n_clusters: desired number of clasters
    :param n_jobs: use parallel with -1
    :param viz: True to output scatterplot
    :param allcolumns: cluster all the columns ( e.g for index -4 use -4,-3,-2,-1)
    :param group: cluster also all the columns together
    :param index: starting index for the dataset columns to use
    :param minibatch: use faster minibatch for large datasts
    :return: a list with the classifiers
    """

    # number of clustes must be <= the dataset rows
    if df.shape[0]<n_clusters:  n_clusters=df.shape[0]

    km = []

    if allcolumns:
        # cluster the 4 separate columns
        for i in range(index, 0):
            j = i + 1 if i != -1 else len(df.columns) # to avoid crash when index is -1
            data = df.iloc[:, i:j]
            X = data.values
            kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(X) if minibatch else KMeans(n_clusters=n_clusters,
                                                                                           n_jobs=n_jobs).fit(X)

            km.append(kmeans)

            # visualize
            if viz: vis.visualize(kmeans, x, y, data.columns[0])

    else: #single column

        j = index + 1 if index != -1 else len(df.columns)  # to avoid crash when index is -1
        data = df.iloc[:, index:j]
        X = data.values
        kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(X) if minibatch else KMeans(n_clusters=n_clusters,
                                                                                        n_jobs=n_jobs).fit(X)
        km.append(kmeans)

        # visualize
        if viz: vis.visualize(kmeans, x, y, data.columns[0])

    if group:
        # cluster with the 4 columns
        X = df.iloc[:, index:].values
        kmeans = MiniBatchKMeans(n_clusters=n_clusters).fit(X) if minibatch else KMeans(n_clusters=n_clusters,
                                                                                        n_jobs=-1).fit(X)
        km.append(kmeans)

        if viz: vis.visualize(kmeans, x , y , "4 grades")

    return km
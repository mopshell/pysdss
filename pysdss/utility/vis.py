# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        vis.py
# Purpose:     utilities for visualization
#
# Author:      claudio piccinini
#
# Updated:     23/02/2017
#-------------------------------------------------------------------------------

import matplotlib.pyplot as plt

def visualize(kmeans, x,y, title):
    """
    Visualize a scatter plot for a classified dataset
    :param kmeans: trained classifier
    :param x: array with coordinates
    :param y: array with coordinates
    :param title: chart title
    :return:
    """
    plt.scatter(x, y, s=10, c=kmeans.labels_)
    plt.title(title)
    plt.show()
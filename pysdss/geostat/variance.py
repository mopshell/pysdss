# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        variance.py
# Purpose:     code for semivariance and covariance
#
#
# Author:      claudio piccinini, based on the code book GIS Algorithms by Ningchuan Xiao
#
# Updated:     27/05/2017
#-------------------------------------------------------------------------------


import numpy as np
from math import sqrt


def distance(a, b):
    """Computes distance between points a and b
    :param a: a list of [X, Y]
    :param b: a list of [X, Y]
    :return: distance between a and b
    """
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def semivar(data, lags, hh):
    """ Calculate empirical semivariance
    :param data: a list or 2-D NumPy array, where each element has x,y,value
    :param lags: distance bins in 1-D array
    :param hh: half of the bin size in distance
    :return: A 2-D array of [[h,h,h,...]
                            [gamma(h),gamma(h),gamma(h),...]]
    """

    semivariance = []
    n = len(data)
    # we calculate the distances between pair of points
    dist = [[distance(data[i][0:2], data[j][0:2]) for i in range(n)] for j in range(n)]
    for h in lags:
        gammas = []
        for i in range(n):
            for j in range(n):
                #if dist[i][j] >= h-hh and dist[i][j]<=h+hh:
                if h-hh <= dist[i][j] <= h+hh: # we check the distances to fill in distance bins
                    gammas.append((data[i][2]-data[j][2])**2)
        if len(gammas) == 0:
            gamma = 0
        else:
            gamma = np.sum(gammas) / (len(gammas)*2.0) #and we calculate the semivariance for this bin
        semivariance.append(gamma)
    semivariance = [[lags[i], semivariance[i]] for i in range(len(lags)) if semivariance[i] > 0]

    return np.array(semivariance).T


def covar(data, lags, hh):
    """Calculates empirical covariance from data
    :param data: a list or 2-D NumPy array, where each element has x,y,value
    :param lags: distance bins in 1-D array
    :param hh: half of the bin size in distance
    :return: A 2-D array of [[h,h,h,...]
                            [c(h),c(h),c(h),...]]
    """

    covariance = []
    n = len(data)
    # we calculate the distances between pair of points
    dist = [[distance(data[i][0:2], data[j][0:2]) for i in range(n)] for j in range(n)]
    for h in lags:
        c = []
        mu = 0
        for i in range(n):
            for j in range(n):
                #if dist[i][j] >= h-hh and dist[i][j]<=h+hh:
                if h - hh <= dist[i][j] <= h + hh:  # we check the distances to fill in distance bins
                    c.append(data[i][2] * data[j][2])
                    mu += data[i][2] + data[j][ 2]
        if len(c) == 0:
            ch = 0
        else:
            mu = mu/(2*len(c))
            ch = np.sum(c) / len(c) - mu*mu #and we calculate the covariance for this bin
        covariance.append(ch)
    covariance = [[lags[i], covariance[i]] for i in range(len(lags))]

    return np.array(covariance).T
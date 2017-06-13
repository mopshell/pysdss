# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        kriging.py
# Purpose:     code for ordinary kriging,simple kriging, indicator
#
#
# Author:      claudio piccinini, based on the code book GIS Algorithms by Ningchuan Xiao
#
# Updated:     29/05/2017
#-------------------------------------------------------------------------------

#TODO cythonize this code!
import warnings


import numpy as np
from pysdss.geostat.variance import distance
from scipy.spatial.distance import cdist


def ordinary(data, model):
    """ Ordinary kriging.
    :param data: an array of [x, y, value, distance to x0]
    :param model: the name of fitted semivariance model (spherical linear gaussian exponential)
    :return: estimated value at x0,standard error,estimated mean,weights
    """


    #debugsolve=False
    #debugweights=False

    # number of points
    n = len(data)
    # get [gamma(xi, x0)]
    k = model(data[:, 3])
    k = np.matrix(k).T  # k is a 1xN matrix
    # add a new row of 1
    k1 = np.matrix(1)
    k = np.concatenate((k, k1), axis=0)

    #K =[[distance(data[i][0:2], data[j][0:2]) for i in range(n)] for j in range(n)]
    #K = np.array(K)                     # list -> NumPy array
    K = cdist(data[:,:2], data[:,:2]) #use scipy instead

    K = model(K.ravel())                # [gamma(xi, xj)]
    K = np.matrix(K.reshape(n, n))      # array -> nxn matrix
    ones = np.matrix(np.ones(n))        # nx1 matrix of 1s

    K = np.concatenate((K, ones.T), axis=1) # add a col of 1s
    ones = np.matrix(np.ones(n+1))          # (n+1)x1 of 1s
    ones[0, n] = 0.0                        # last one is 0
    K = np.concatenate((K, ones), axis=0)   # add a new row
    try:
        w = np.linalg.solve(K, k)           # solve: K w = k
    except:
        #debugsolve=True
        w = np.linalg.lstsq(K, k)[0]        # solve: K w = k with  least-squares in case of singular matrix
        #print(w[:-1].sum(), end=" ", sep=" ")
    ############################
    #if (w<0).any:
        #w[w<0]=0
        #debugweights=True
        #print("negative weights -> output " + str(w))
        #warnings.warn("some kriging weiths are negative!", RuntimeWarning)

    ############################

    zhat = (np.matrix(data[:, 2]) * w[:-1])[0, 0] # est vlaue
    sigmasq = (w.T * k)[0, 0]               # est error var
    if sigmasq < 0:
        sigma = 0
    else:
        sigma = np.sqrt(sigmasq)            # error

    #if debugsolve:print("singular matrix -> output " + str(zhat))
    #if debugweights:print("negative weights -> output " + str(zhat))


    return zhat, sigma, w[-1][0], w         # est, err, mu, w


def simple(data, mu, model):
    """Simple kriging
    :param data: an array of [X, Y, Val, Distance to x0]
    :param mu: mean of the data
    :param model: the name of fitted semivariance model (spherical linear gaussian exponential)
    :return: the estimated value at the target location,standard error,weights
    """

    # number of points
    n = len(data)
    # get [gamma(xi, x0)]
    k = model(data[:, 3])
    k = np.matrix(k).T  # 1xN matrix

    #K = [[distance(data[i][0:2], data[j][0:2]) for i in range(n)] for j in range(n)]
    #K = np.array(K)                     # list -> NumPy array
    K= cdist(data[:,:2], data[:,:2]) #use scipy instead

    K = model(K.ravel())                # [gamma(xi, xj)]
    K = np.matrix(K.reshape(n, n))      # array -> nxn matrix

    try:
        w = np.linalg.solve(K, k)           # solve: K w = k
    except:
        w = np.linalg.lstsq(K, k)[0]        # solve: K w = k with  least-squares in case of singular matrix

    R = data[:, 2] - mu                     # get residuals
    zhat = (np.matrix(R)*w)[0, 0]       # est residual
    zhat = zhat + mu                    # est value
    sigmasq = (w.T*k)[0, 0]             # est error variance
    if sigmasq < 0:
        sigma = 0
    else:
        sigma = np.sqrt(sigmasq)        # error
    return zhat, sigma, w               # est, error, weights


def indicator():
    raise NotImplementedError("indicatore kriging is not implemented!")
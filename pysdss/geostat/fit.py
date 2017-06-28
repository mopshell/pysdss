# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        fit.py
# Purpose:     code for fitting semivariogram
#
#
# Author:      claudio piccinini, based on the code book GIS Algorithms by Ningchuan Xiao
#
# Updated:     27/05/2017
#-------------------------------------------------------------------------------

import numpy as np


@np.vectorize
def spherical(h, c0, c, a):
    """ Calculate theoretical semivariogram at distance h for spherical function
    :param h: distance
    :param c0: nugget
    :param c: sill
    :param a: range
    :return: theoretical semivariogram at distance h
    """
    if h <= a:
        return c0 + c*(3.0*h/(2.0*a) - ((h/a)**3.0)/2.0)
    else:
        return c0 + c


@np.vectorize
def gaussian(h, c0, c, a):
    """Theoretical semivariogram at distance h for gaussian function
    :param h: distance
    :param c0: nugget
    :param c: sill
    :param a: range
    :return:  theoretical semivariogram at distance h
    """
    return c0 + c*(1-np.exp(-h*h/((a)**2)))


@np.vectorize
def exponential(h, c0, c, a):
    """Theoretical semivariogram at distance h for exponential function
    :param h: distance
    :param c0: nugget
    :param c: sill
    :param a: range
    :return:  theoretical semivariogram at distance h
    """
    #return c0 + c*(1-np.exp(-h/a))
    s = h / a
    return c0 + c*(1-np.exp(-s))


@np.vectorize
def linear(h, c0, c, a):
    """Theoretical semivariogram at distance h for linear function
    :param h: distance
    :param c0: nugget
    :param c: sill
    :param a: range
    :return:  theoretical semivariogram at distance h
    """
    if h <= a:
        return c0 + c*(h/a)
    else:
        return c0 + c


@np.vectorize
def power(h, c0, c, a, p):
    """Theoretical semivariogram at distance h for power function
    :param h: distance
    :param c0: nugget
    :param c: sill
    :param a: range
    :param p: power parameter between 0 and 2
    :return:  theoretical semivariogram at distance h
    """

    if not 0<= p <= 2: raise ValueError("power function expects a power parameter between 0 and 2")
    return c0 + c * (h**p)


def fitsemivariogram(data, semivar, model, numranges=200,forcenugget=False):
    """Fits a theoretical semivariance model.
    :param data: data, NumPy 2D array, each row has (X, Y, Value)
    :param semivar: empirical semivariances as a 2-D array of [[h,h,h,...]
                                                                [gamma(h),gamma(h),gamma(h),...]]
    :param model: one of the semivariance models: spherical, Gaussian, exponential, and linear
    :param numranges: number of ranges to test
    :param forcenugget: True if we want to srat the fitted line with the nugget
    :return: a lambda function that serves as a fitted model of semivariogram.
            This function will require one parameter(distance).
    """

    c = np.var(data[:, 2])          # sill

    if forcenugget:
        if semivar[0][0] is not 0.0:      # cnugget
            semivar[0][0] = 0.0
        c0 = semivar[1][0]
    else:
        if semivar[0][0] is not 0.0:      # cnugget
            c0 = 0.0
        else:
            c0 = semivar[0][1]
    minrange, maxrange = semivar[0][1], semivar[0][-1]
    ranges = np.linspace(minrange, maxrange, numranges)
    #calculate the error for all the ranges and get the range with lower error
    errs = [np.mean((semivar[1] - model(semivar[0], c0, c, r)) ** 2)for r in ranges]
    a = ranges[errs.index(min(errs))]  # optimal range

    return lambda h: model(h, c0, c, a)
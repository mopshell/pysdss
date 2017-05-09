# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        compensatory.py
# Purpose:     functions for compensatory and noncompensatory constraints
#
#
# Author:      claudio piccinini
#
# Updated:     11/04/2017
#-------------------------------------------------------------------------------

import numpy as np

###NONCOMPENSATORY
def noncompensatory(layer, thresholds, kind):
    """
    Apply minimum acceptable thresholds to layers and compute AND logical operator
    :param layer: a list of numpy arrays
    :param thresholds: list of thresholds for the layers
    :param kind: specify "conjunctive" or "disjunctive" screening
    :return: a boolean array for the feasible,infeasible alternatives
    """
    if kind not in ["conjunctive", "disjunctive"]:
        raise ValueError("Compensatory screening shuld be 'conjunctive' or 'disjunctive'")

    for i, l in enumerate(layer):
        l[l < thresholds[i]] = 0
        l[l >= thresholds[i]] = 1

    if kind == "conjunctive":
        return np.logical_and.reduce(layer)
    else:
        return np.logical_or.reduce(layer)

###COMPENSATORY
def compensatory(layer, formula, cutoff):
    """
    Apply an objective function to the layers and a cutoff value
    
    NOTE: Layers in the formula must be written as layers[0], layers[1],...
    
    :param layer: a list of numpy arrays
    :param formula: the string representation of the objective function
    :param cutoff: the cutoff value
    :return: an array for the feasible (1) and infeasible alternatives (0)
    """
    x = eval(formula)
    x[x < cutoff] = 0
    x[x >= cutoff] = 1
    return x

if __name__ == "__main__":

    print("#######NONCOMPENSATORY######")
    a = np.array([8,2,2,10]) #  1001
    b = np.array([9,8,0,12]) #  1101
    c = np.array([3,8,9,18]) #  0001
    print(noncompensatory([a,b,c], [5, 8, 15], "conjunctive"))  #FFFT
    a = np.array([8,2,2,10])
    b = np.array([9,8,0,12])
    c = np.array([3,8,9,18])
    print(noncompensatory([a,b,c], [5, 8, 15], "disjunctive"))  #TTFT

    print("#######COMPENSATORY######")
    a = np.array([8,2,2,10]) #  1001
    b = np.array([9,8,0,12]) #  1101
    c = np.array([3,8,9,18]) #  0001

    f = "layer[0] + 5*layer[1] - layer[2]" #50 34 -7 52
    print(compensatory([a, b, c], f, 5))#1101






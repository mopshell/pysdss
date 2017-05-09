# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:        function.py
# Purpose:     scale criterion maps with utility/value functions
#
#
# Author:      claudio piccinini
#
# Updated:     11/04/2017
# -------------------------------------------------------------------------------

import numpy as np


def value_function(x, function):
    """
    Transform criteria scores through a value curve. This method is for decision under certainty
    :param x: the raw criterion score in numpy array format 
    :param function: textual refresentation of the function, x must be the variable name
    :return: the standardized criterion score
    """
    if "x" not in function:
        raise ValueError("The value function should contain the variable x")
    return eval(function)


def utility_function(x):
    """
    Transform criteria scores through a utility curve. This method is for decision under uncertainty
    :param x:  the raw criterion score in numpy array format  
    :return: the standardized criterion score
    """
    raise NotImplementedError("The function utility_function is not implemented")

if __name__ == "__main__":

    x = np.array([1, 2, 4, 5, 7, 8])
    print(x)
    print("#######################")
    print("VALUE FUNCTION")
    f="0.001*x"
    print(value_function(x,f))

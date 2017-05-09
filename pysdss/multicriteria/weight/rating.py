# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:        rating.py
# Purpose:     criterion weighting with rating methods
#
#
# Author:      claudio piccinini
#
# Updated:     11/04/2017
# -------------------------------------------------------------------------------


def ratio_estimation(ratio):
    """
    Ratio estimation procedure (Easton,1973)
    see Malczewski,1999, p. 181
    :param ratio: A list with the ratio score, the most important criterion must have value 100
    :return: 
    """
    if 100 not in ratio or max(ratio)>100:
        raise ValueError("The most important criterion must have value 100")

    # define weights
    m= min(ratio)
    ratio = [ i/m for i in ratio]
    # normalize weights
    total = sum(ratio)
    return [i/total for i in ratio]


if __name__ == "__main__":

    print("#######ratio_estimation######")

    r = [50, 75, 10, 100, 63]
    print(ratio_estimation(r))

    try:
        r = [50, 75, 10, 110, 63]
        print(ratio_estimation(r))
    except ValueError as e:
        print(e)
    try:
        r = [50, 75, 10, 90, 63]
        print(ratio_estimation(r))
    except ValueError as e:
        print(e)

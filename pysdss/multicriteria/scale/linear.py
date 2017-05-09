# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        linear.py
# Purpose:     scale criterion maps with linear scale transformation
#

#
# Author:      claudio piccinini
#
# Updated:     11/04/2017
#-------------------------------------------------------------------------------


import numpy as np


def maximum_score(raw_score, goal, both=False):
    """
    Linear transformation with maximum score   
    see Malczewski,1999, p. 117     
    :param raw_score: a numpy array with the raw criterion scores
    :param goal:   can be "maximize" for benefit criteria or "minimize" for cost criteria
    :param both: set true for decision with both benefit and cost criteria (Hwang and Yoon, 1981)
    :return: The standardized criterion score
    """,x
    if goal not in ["maximize", "minimize"]:
        raise ValueError("Linear transformation goal can be 'maximize' or 'minimize' ")
    if goal == "minimize":  #cost criteria , the lower the score the better the performance
        if both:  # see Tzeng and Huang, 2011 Multiple attribute decision Making  p.58
            return np.min(raw_score)/raw_score
        else:
            return 1 - (raw_score/np.max(raw_score))
    else:  # benefit criteria. the higher the cost the better the performance
            return raw_score/np.max(raw_score)


def score_range(raw_score, goal):
    """
    Linear transformation with score range
    see Malczewski,1999, p. 118
    :param raw_score: 
    :param goal: can be "maximize" for benefit criteria or "minimize" for cost criteria
    :return: the standardized criterion score
    """
    if goal not in ["maximize", "minimize"]:
        raise ValueError("Linear transformation goal can be 'maximize' or 'minimize' ")
    delta = np.max(raw_score)-np.min(raw_score)
    if goal == "minimize":
        return (np.max(raw_score)-raw_score)/delta
    else:
        return (raw_score-np.min(raw_score))/delta


if __name__ == "__main__":

    x = np.array([1, 2, 4, 5, 7, 8])
    print(x)
    print("#######################")
    print("MAXIMUM SCORE")
    print("maximize",maximum_score(x, "maximize"))
    print("minimize",maximum_score(x, "minimize"))
    print("maximize with both", maximum_score(x, "maximize", both=True))
    print("minimize with both", maximum_score(x, "minimize", both=True))
    print("#######################")
    print("SCORE RANGE")
    print("maximize",score_range(x, "maximize"))
    print("minimize",score_range(x, "minimize"))


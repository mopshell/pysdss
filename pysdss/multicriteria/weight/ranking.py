# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:        ranking.py
# Purpose:     criterion weighting with ranking methods
#
#
# Author:      claudio piccinini
#
# Updated:     11/04/2017
# -------------------------------------------------------------------------------

import math


def rank_sum(rank):
    """
    Calculate normalized weights
    :param rank: a list of criterion ranks 
    :return: list of normalized weights
    """
    total = sum([len(rank)-i+1 for i in rank])
    return [(len(rank)-i+1)/total for i in rank]


def rank_reciprocal(rank):
    """
    Calculate rank_reciprocal weights
    :param rank: a list of criterion ranks 
    :return: list of normalized weights
    """
    total = sum([1/i for i in rank])
    return [(1/i)/total for i in rank]


def rank_exponent(rank,exponent):
    """
    Calculate rank_exponent weights   
    :param rank: a list of criterion ranks 
    :param exponent: for exponent = 0 all weights are the same, 
                    for exponent = 1 the output is rank_sum
                    for exponent>1 weights get steeper
    :return: normalized weights
    """
    total = sum([math.pow(len(rank)-i+1,exponent) for i in rank])
    return [math.pow(len(rank)-i+1,exponent)/total for i in rank]

if __name__ == "__main__":

    rank = [1,2,3,4]
    print("ranks",rank)
    print("#####RANK SUM####")
    print(rank_sum(rank))
    print(sum(rank_sum(rank)))
    print("#####RANK RECIPROCAL####")
    print(rank_reciprocal(rank))
    print(sum(rank_reciprocal(rank)))
    print("#####RANK EXPONENT####")
    print("#####RANK RECIPROCAL####")
    print("exponent 0",rank_exponent(rank,0))
    print("exponent 1",rank_exponent(rank,1))
    print("exponent 2",rank_exponent(rank,2))
    print("exponent 2 sum",sum(rank_exponent(rank,2)))

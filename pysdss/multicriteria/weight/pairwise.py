# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:        pairwise.py
# Purpose:     criterion weighting with pairwise comparison
#
#
# Author:      claudio piccinini
#
# Updated:     11/04/2017
# -------------------------------------------------------------------------------
import numpy as np


def get_consistency_ratio(n):
    """
    Get the random inconsistency index for the current number of evaluation criteria
    :param n: the number of evaluation criteria
    :return: the random  
    """
    #random inconsistency indices (Saaty, 1980)
    ri = {1:0,2:0,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49,11:1.51,12:1.48,13:1.56,
    14:1.57,15:1.59}

    return ri[n]


def pairwise_comparison(matrix):
    """
    Pairwise comparison for the analytic hierarchy process(Saary,1980) 
    return consistency_ratio and a list with criterion_weights
    
    consitency ratio must be <0.10 otherwise the pairwise comparison matrix must be recalculated
    
    :param matrix: The pairwise comparison matrix of the evaluation criteria as a numpy array
    
    The pairwise comparison matrix scores should follow this table
    
    1   Equal importance
    2   Equal to moderate importance 
    3   Moderate importance
    4   Moderate to strong importance
    5   Strong importance
    6   Strong to very strong importance
    7   Very strong importance
    8   Very to extremely strong importance
    9   Extreme importance
    
    :return: consistency_ratio and a list with criterion_weights
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The method needs a square matrix")
    n_elements = matrix.shape[0]
    if n_elements < 3 or n_elements > 15:
        raise ValueError("The pairwise comparison matrix needs 3 to 15 elements")

    original_matrix = np.copy(matrix)

    #####CRITERION WEIGHTS########
    # step1 calculate sum of columns
    column_total = [np.sum(matrix[:,i]) for i in range(matrix.shape[1])]
    # step2 normalized pairwise comparison matrix
    for i, s in enumerate(column_total):
        matrix[:, i] = matrix[:, i]/s
    #step3 compute criterion weights as average values by row
    criterion_weights=[np.sum(matrix[i, :])/matrix.shape[1] for i in range(matrix.shape[1])]

    #####CONSISTENCY RATIO#####
    # step1
    for i, s in enumerate(criterion_weights):
        original_matrix[:, i] = original_matrix[:, i]*s
    sum_by_row = np.sum(original_matrix,axis=1)
    # step2
    consistency_vector = sum_by_row/np.array(criterion_weights)
    lmbda = np.mean(consistency_vector)
    consistency_index = (lmbda - n_elements)/ (n_elements-1)
    consistency_ratio = consistency_index/get_consistency_ratio( n_elements)

    return consistency_ratio, criterion_weights

if __name__ == "__main__":
    print("#######PAIRWISE COMPARISON######")

    print("#######INCONSISTENT COMPARISON######")
    pcm = np.array([[1,   4,   7],
                    [1/4, 1,   5],
                    [1/7, 1/5, 1]])
    print(pairwise_comparison(pcm))
    print("#######CONSISTENT COMPARISON######")
    pcm = np.array([[1,   4,   7],
                    [1/4, 1,   4],
                    [1/7, 1/4, 1]])
    print(pairwise_comparison(pcm))
    print("#######WRONG COMPARISON######")
    try:
        pcm = np.array([[1,  4],
                        [1/4,1]])
        print(pairwise_comparison(pcm))
    except ValueError as v:
        print(v)
    print("#######WRONG COMPARISON######")
    try:
        pcm = np.array([[1, 4, 7, 1],
                        [1 / 4, 1, 4,3],
                        [1 / 7, 1 / 4, 1,4]])
        print(pairwise_comparison(pcm))
    except ValueError as v:
        print(v)








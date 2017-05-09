# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        main.py
# Purpose:     on disk filtering
#
# Author:      claudio piccinini
#
# Updated:     23/02/2017
#-------------------------------------------------------------------------------

#TODO: try...except is missing

import pandas as pd


def calculate_stat(df, nstd=1, colname=""):
    """
    Calculate mean, std, and min max values
    :param df: pandas dataframe
    :param nstd: number of standard deviations use to calculate min and max
    :param colname: columns to use
    :return: mean, std, min , max
    """
    mean = df[colname].mean()
    std = df[colname].std()
    mn = mean - (std * nstd)
    mx = mean + (std * nstd)
    return mean, std, mn, mx

def calculate_stat_byrow(df,nstd=1, rowname=None, colname=None):
    """
    Claculate statistics for each row in a dataframe.
    The dataframe should have a field with the row number
    The user need to pass a field name to calculate the statistics
    :param df: dataframe
    :param nstd: number of standard deviations to calculate the min max values (default 1)
    :param rowname: field with the row number
    :param colname: field for statistics calculation
    :return: the filed statistics  average, standard deviation,
    minimum and max value based on a custom number of standard deviations
    """

    if (not rowname) or (not colname):
        raise Exception("Please pass a row field and a field name")
    stats = []
    # find unique row values
    rows = df[rowname].unique()
    for row in rows:
        # select one vineyard row
        fdf = df[df[rowname] == row]
        mean, std, mn, mx = calculate_stat(fdf, nstd=nstd, colname=colname)
        stats.append({row : [mean, std, mn, mx]})
    return stats


def set_field_value(df, value=None, colname=""):
    """
    Update the value for a pandas dataframe column
    :param df: pandas dataframe
    :param value: value to set
    :param colname: field to update
    :return:
    """
    df[colname] = value


def filter_bystep(df, step=1, rowname=None):
    """
    Reduce the number of points using a step
    Pass a column with the row name to filter by row
    :param df: dataframe
    :param step: filter step
    :param rowname: column with row number
    :return: 2 dataframes, one for values to keep and another with values to discard
    """

    if rowname: # TODO: is this useful?
        keepframes = []
        discardframes = []
        #frames = []
        # find unique row numbers
        rows = df[rowname].unique()

        for row in rows:
            # select one vineyard row
            fdf = df[df[rowname] == row]

            # get number of points for this row
            npoints = fdf.shape[0]

            # subset the row with a step
            #frames.append(fdf.iloc[range(0, npoints, step), :])

            keepframes.append(fdf[fdf.index.isin(range(0, npoints, step))])
            discardframes.append(fdf[~fdf.index.isin(range(0, npoints, step))])

        # concatenate rows
        return pd.concat(keepframes),pd.concat(discardframes)

    else:
        # get number of points for this row
        npoints = df.shape[0]
        # return df.iloc[range(0, npoints, step), :]
        keep = df[df.index.isin(range(0, npoints, step))]
        discard = df[~df.index.isin(range(0, npoints, step))]
        return keep, discard


def filter_byvalue(df, value=0.8, operator=">=", colname=None,rowname=None ):
    """
    Reduce the number of points using a threshold
    values >=  threshold will be selected by default, can be also >,<=,<
    Pass a column with the row name to filter by row
    :param df: dataframe
    :param value: threshold value
    :param operator: threshold direction, default select values >= threshold; can be also >,<=,<
    :param colname: column to filter
    :param rowname: column with row number
    :return: 2 dataframes, one for values to keep and another with values to discard
    """

    if rowname:  # TODO: is this useful?
        keepframes = []
        discardframes = []
        #frames = []
        # find unique row values
        rows = df[rowname].unique()

        for row in rows:
            # select one vineyard row
            fdf = df[df[rowname] == row]

            # subset the row with a value and append to list
            if operator == ">=":
                keepframes.append(fdf[fdf[colname] >= value])
                discardframes.append(fdf[fdf[colname] < value])
            elif operator == ">":
                keepframes.append(fdf[fdf[colname] > value])
                discardframes.append(fdf[fdf[colname] <= value])
            elif operator == "<=":
                keepframes.append(fdf[fdf[colname] <= value])
                discardframes.append(fdf[fdf[colname] > value])
            elif operator == "<":
                keepframes.append(fdf[fdf[colname] < value])
                discardframes.append(fdf[fdf[colname] >= value])

        # concatenate rows
        #return pd.concat(frames)
        return pd.concat(keepframes), pd.concat(discardframes)

    else:
        # subset the row with a value and append to list
        if operator == ">=": return df[df[colname] >= value], df[df[colname] < value]
        elif operator == ">": return df[df[colname] > value], df[df[colname] <= value]
        elif operator == "<=": return df[df[colname] <= value], df[df[colname] > value]
        elif operator == "<": return df[df[colname] < value], df[df[colname] >= value]
        else:
            print("Incorrect Operator: use '>=', '>', '<=', '<', returning full keep dataframe and empty discard dataframe")
            return df, df.truncate(after=-1)


def filter_bystd(df, nstd=1, colname=None,rowname=None, backstats=False):
    """
    Reduce the number of points using an average and standard deviation
    Pass a column with the row name to filter by row

    note: the discard dataframe mey not be ordered, order it after returning

    :param df: dataframe
    :param std: how many standard deviation I want? 1,2,3?
    :param colname: columns to filter
    :param rowname: column with row number
    :param backstats: return statistics back
        as a list mean, std, mn, mx
        as a list of lists row, mean, std, mn, mx
    :return: 2 dataframes, one for values to keep and another with values to discard
    """


    if rowname:
        stats = []
        keepframes = []
        discardframes = []

        # find unique row values
        rows = df[rowname].unique()

        for row in rows:
            # select one vineyard row
            fdf = df[df[rowname] == row]

            mean, std, mn, mx = calculate_stat(fdf, nstd, colname)
            if backstats: stats.append([row,mean, std, mn, mx])
            # subset the row with a value and append to list
            z=fdf[fdf[colname] >= mn]
            keepframes.append(z[z[colname] <= mx]) # TODO: how to concatenate expressions?
            discardframes.append(pd.concat( [fdf[fdf[colname]< mn],fdf[fdf[colname] > mx]]))

        # concatenate rows
        if backstats:
            return pd.concat(keepframes), pd.concat(discardframes), stats
        else:
            return pd.concat(keepframes), pd.concat(discardframes)

    else:
        mean, std, mn, mx = calculate_stat(df, nstd, colname)
        # subset the row with min and max values
        z = df[df[colname] >= mn] # TODO: how to concatenate expressions?
        if backstats:
            return z[z[colname] <= mx],pd.concat( [df[df[colname]< mn],df[df[colname] > mx]]), [mean, std, mn, mx]
        else:
            return z[z[colname] <= mx],pd.concat( [df[df[colname]< mn],df[df[colname] > mx]])
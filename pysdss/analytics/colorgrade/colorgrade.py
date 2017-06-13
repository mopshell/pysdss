# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        colorgrade.py
# Purpose:     code for colorgrade analytics
#               this is for the workflow 2, refer to the documentation
#
# Author:      claudio piccinini
#
# Created:     04/04/2017
#-------------------------------------------------------------------------------

import pandas as pd
import math
import json
import shutil
import warnings

import os
import os.path
from osgeo import gdal
#from osgeo import gdalconst as gdct
#from osgeo import ogr
#from osgeo import osr

import numpy as np
#from scipy.interpolate import Rbf
#from scipy.misc import imsave

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from matplotlib import cm


#from rasterstats import point_query

from pysdss.utility import fileIO
from pysdss.utility import gdalRasterIO as gdalIO

from pysdss import utils
from pysdss.database import database as dt
from pysdss.filtering import filter
from pysdss.gridding import gridding as grd
from pysdss.utility import utils as utils2


#################berrycolor_workflow

def interpret_result(stats, folder):
    """
    Interpret the dictionary with the results from berrycolor_workflow and save a table

    table columns are:
    row|avg|std|area|nozero_area|zero_area|raw_count|raw_avg|raw_std|visible_count|visible_avg|visible_std

    :param stats:
        dictionary in the form

        {"colorgrade": {row: [0.031, 0.029, 93.0, 83.75, 9.25, 118339.1, 318.11, 29.281, 213405.12, 573.66968, 61.072674],
        
        the list contains
        average, std, totalarea,total nonzero area, total zeroarea, total raw fruit count, average raw fruit count, 
        std raw fruit count ,total visible fruitxm, average visible fruitXm, std visible fruitXm

    :return: None
    """

    # get the number of colors (this will be the number of output files)
    ncolors = len(stats)  # 4
    colors = [i for i in stats]  # a,b,c,d

    # get the number of rows
    nrows = len(stats[colors[0]])

    # check if rows start with 0 or 1, this will be needed to fill in the array
    rowfix = 1
    if 0 in stats[colors[0]]: rowfix = 0

    # get the number of columns
    # define the column names and convert to string
    colmodel = ["avg", "std", "area", "nozero_area", "zero_area", "raw_count", "raw_avg", "raw_std", "visible_count",
                "visible_avg", "visible_std"]
    cols = ["row"] + colmodel
    NCOL = len(cols)
    cols = ','.join(cols)

    # create a numpy array to store result
    output = np.zeros((nrows, NCOL), dtype=np.float32)

    # iterate by color
    for color in colors:
        for row in stats[color]:
            row=int(row)#be sure this is a number
            output[row - rowfix, :] = [row] + stats[color][row]  # select the array row to fill in based on the row number
        # save array to disk as a csv for this color grade
        np.savetxt(folder + "/_" + color + "_output.csv", output, fmt='%.3f', delimiter=',', newline='\n', header=cols,
                   footer='', comments='')
        output.fill(0)


def get_counts(stats, rlength, folder, threshold=0.05):
    """
    return a table with the predicted number of berries for each row and the numbed of berries per color grade
    
    table columns are:
    row|length|total_berries|a_%|b_%|c_%|d_%|a|b|c|d where "total_berries" is the predicted total number of berries; 
    "a,b,c,d" are the predicted percentages and numbers
    for each color grade
    
    :param stats: dictionary in the form

        {"colorgrade": {row: [0.031, 0.029, 93.0, 83.75, 9.25, 118339.1, 318.11, 29.281, 213405.12, 573.66968, 61.072674],
        the list values are average, std, totalarea,total nonzero area, total zeroarea, total raw fruit count, 
        average raw fruit count, std raw fruit count ,total visible fruitxm, average visible fruitXm, std visible fruitXm
    :param rlenght: the array with row_id - length,  create with utils.row_length
    :param folder: outputfolder
    :param threshold: threshold  to check if the sum of the color grade percentages is equal to 1, default 5%
    :return: 
    """

    # get the color grades
    # ncolors = len(stats)  # 4
    colors = [i for i in stats]  # a,b,c,d
    colors.sort()

    # check the number of rows are the same
    if len(stats[colors[0]]) != rlength.shape[0]:
        raise ValueError("the number of rows in stats and rlength are different!")
    # check the row ids are the same
    if sorted(list(stats[colors[0]].keys())) != sorted(list(rlength[:,0])):
        raise ValueError("the rowid in stats and rlength are different!")

    # create a numpy array to store result
    output = np.zeros((rlength.shape[0], 3+len(colors)*2), dtype=np.float32)
    # initialize file header
    cols = ["row_id", "length","total_berries"]
    [cols.append(j+"_%") for j in colors]
    [cols.append(j) for j in colors]

    row = None
    length = None
    visible_fruit_m = None
    for i in range(rlength.shape[0]):

        row = output[i,0] = rlength[i,0]
        length = output[i,1] = rlength[i,1]
        visible_fruit_m = stats[colors[0]][row][-2]
        tot_visible_fruit = output[i,2] = length*visible_fruit_m

        # check percentages and fix them
        clr = [stats[c][row][0] for c in colors] # this contains the percentages for the color grades
        diff = 1.0 - sum(clr)  # in theory the sum should be 1, but sometimes is not....
        #print(diff)
        if abs(diff) > threshold:  # if the difference is more than the threshold fix percentages:
            y = [abs(diff) * x for x in clr]
            # print(clr, sum(clr))
            if diff > 0:
                clr = list(map(lambda x, y: x + y, clr, y))
            else:
                clr = list(map(lambda x, y: x - y, clr, y))
            # print(clr,sum(clr))

        # use percentages to calculate the output values
        for j, x in enumerate(colors):
            output[i, 3+j] = clr[j]
            output[i, 3+len(colors)+j] = clr[j] * tot_visible_fruit


    np.savetxt(folder + "/_" + "count_output.csv", output, delimiter=',', fmt='%.3f', newline='\n', header=','.join(cols),footer=''
                   , comments='')


def get_boxes(file, berry_x_unit, box_volume):
    """
    Calculate the number of boxes for each row and output a csv file
    :param file: the file with the counts per row 
    :param berry_x_unit: the average number of berries * unit of volume 
    :param box_volume: the volume of 1 box
    :return: 
    """

    #1 open the count file
    if not os.path.exists(file):  raise FileNotFoundError("the file "+file+" does not exist!")
    df = pd.read_csv(file)
    cols= df.columns
    cols=list(cols) + ['berries_xunit','box_volume', 'n_boxes']
    #2 initialize empty array with 3 new fields (berries_xunit,box_volume, n_boxes)
    out = np.zeros((df.shape[0],df.shape[1]+3))
    #copy file to new file
    out[:,:-3] = df.values
    #fill in new file with 2 columns and calculate the n_boxes
    out[:,-3] = berry_x_unit
    out[:,-2] = box_volume
    out[:,-1] = (df['total_berries'].values/berry_x_unit)/box_volume
    #save file to disk
    folder = os.path.dirname(file)
    np.savetxt(folder + "/_" + "count_boxes_output.csv", out, delimiter=',', fmt='%.3f', newline='\n', header=','.join(cols),footer=''
                   , comments='')


def berrycolor_workflow(id, file, usecols, new_column_names, outfolder, average_point_distance=None, grid=None,
                          rowdirection="x",area_multiplier=2, filterzero=False, project="32611", buffersize=0.25,
                          nodata=-1, force_interpolation=False ):
    """
    This is the workflow for direct interpolation of high and low density colorgrade data
    For high density data statistics are calculated directly from the vector data
    
    The function returns a dictionary with statistics in the form
    {"colorgrade": {row: [0.031, 0.029, 93.0, 83.75, 9.25, 118339.1, 318.11, 29.281, 213405.12, 573.66968, 61.072674],
        the list values are average, std, totalarea,total nonzero area, total zeroarea, total raw fruit count, average raw fruit count, 
        std raw fruit count ,total visible fruitxm, average visible fruitXm, std visible fruitXm
       
    :param id: the unique id for this operation
    :param file: the csv file with the data
    :param usecols: a list with the original field names 
    :param new_column_names: a list with the new field names

        in the current implementation names can only be
        ["id", "row", "raw_fruit_count", "visible_fruit_per_m", "lat", "lon", "a", "b", "c", "d"]

    :param outfolder: the folder for the output 
    :param average_point_distance: this number is use to decide how to rasterize the rows and the search radius when
                                    for the inverse distance is applied
    :param grid: the interpolation grid, if None the grid will be calculated on the data boundaries

        grid format is {'minx': ,'maxx': ,'miny': ,'maxy': ,'delty': , 'deltx': }
    :param rowdirection: the direction of the vineyard rows, default "x", pass any other string otherwise
    :param area_multiplier: used to increase the interpolation grid resolution, use 2 to halve the pixel size
    :param filterzero: True to filter out zeros from the data
    :param project: epsg string for reprojection
    :param buffersize: the buffersize for buffered polyline
    :param nodata: the nodata value to assign to nodata pixels
    :param force_interpolation: True if interpolation should be carried out also for average_point_distance under the threshold
    :return: a dictionary with the statistics in the form

        {"colorgrade": {row: [0.031, 0.029, 93.0, 83.75, 9.25, 118339.1, 318.11, 29.281, 213405.12, 573.66968, 61.072674],
        the list values are average, std, totalarea,total nonzero area, total zeroarea, total raw fruit count, average raw fruit count, 
        std raw fruit count ,total visible fruitxm, average visible fruitXm, std visible fruitXm
    """

    #this is the hardcoded threshold under which data is considered dense
    threshold = 0.5

    #######checking inputs
    if new_column_names != ["id", "row", "raw_fruit_count", "visible_fruit_per_m", "lat", "lon", "a", "b", "c", "d"]:
        raise ValueError('With the current implementation column names must be '
                         '["id", "row", "raw_fruit_count", "visible_fruit_per_m", "lat", "lon", "a", "b", "c", "d"]')
    if filterzero:
        raise NotImplementedError("filtering out zero is not implemented")
    #########

    # set the path to folder with interpolation settings
    jsonpath = os.path.join(os.path.dirname(__file__), '../../..', "pysdss/experiments/interpolation/")
    jsonpath = os.path.normpath(jsonpath)

    # open the file and fix the column names
    df = pd.read_csv(file, usecols=usecols)
    df.columns = new_column_names

    if project:
        print("reprojecting data")

        west, north = utils2.reproject_coordinates(df, "epsg:" + str(project), "lon", "lat", False)
        filter.set_field_value(df, west, "lon")
        filter.set_field_value(df, north, "lat")

        # overwrite the input file
        df.to_csv(file, index=False)  # todo check this is ok when there is no filtering

    if not average_point_distance:
        print("calculating average distance")
        average_point_distance = utils.average_point_distance(file, "lon", "lat", "row", direction=rowdirection,
                                                              rem_duplicates=True, operation="mean")

    #set the interpolation radius based on threshold and average_point distance
    radius = "0.8" if average_point_distance <= threshold else str(average_point_distance * 2)

    print("defining interpolation grid")

    # define the interpolation grid
    if grid is None:
        # need a grid
        minx = math.floor(df['lon'].min())
        maxx = math.ceil(df['lon'].max())
        miny = math.floor(df['lat'].min())
        maxy = math.ceil(df['lat'].max())
        delty = maxy - miny  # height
        deltx = maxx - minx  # width
    else:  # the user passed a grid object
        minx = grid['minx']
        maxx = grid['maxx']
        miny = grid['miny']
        maxy = grid['maxy']
        delty = grid['delty']  # height
        deltx = grid['deltx']  # width

    ''''
    if filterzero: #todo find best way for filtering for multiple columns
        #keep, discard = filter.filter_byvalue(df, 0, ">", colname="d")
    else:
        keep = df
        discard = None
    if discard: discard.to_csv(folder + id + "_discard.csv", index=False)            
    '''

    # open the model for for creting gdal virtual files
    with open(jsonpath + "/vrtmodel.txt") as f:
        xml = f.read()

    ##### define rows

    print("extracting rows")

    if average_point_distance > threshold:  ########SPARSE DATA, create rasterized buffered polyline

        # 1 convert point to polyline
        utils2.csv_to_polyline_shapefile(df, ycol="lat", xcol="lon", linecol="row", epsg=project,
                                         outpath=outfolder + "/rows.shp")
        #  2 buffer the polyline
        utils2.buffer(outfolder + "/rows.shp", outfolder + "/rows_buffer.shp", buffersize)
        #  3rasterize poliline

        path = jsonpath + "/rasterize.json"
        params = utils.json_io(path, direction="in")

        params["-a"] = "id_row"
        # params["-te"]= str(minx) + " " + str(miny) + " " + str(maxx) + " " + str(maxy)
        params["-te"] = str(minx) + " " + str(maxy) + " " + str(maxx) + " " + str(miny)
        params["-ts"] = str(deltx * area_multiplier) + " " + str(delty * area_multiplier)
        params["-ot"] = "Int16"
        params["-a_nodata"] = str(nodata)
        params["src_datasource"] = outfolder + "rows_buffer.shp"
        params["dst_filename"] = outfolder + "/" + id + "_rows.tiff"

        # build gdal_grid request
        text = grd.build_gdal_rasterize_string(params, outtext=False)
        print(text)
        text = ["gdal_rasterize"] + text
        print(text)

        # call gdal_rasterize
        print("rasterizing the rows")
        out, err = utils.run_tool(text)
        print("output" + out)
        if err:
            print("error:" + err)
            raise Exception("gdal rasterize failed")


    else:  #############DENSE DATA   we are under the threshold, rasterize points with nearest neighbour

        if force_interpolation:
            data = {"layername": os.path.basename(file).split(".")[0], "fullname": file, "easting": "lon",
                    "northing": "lat", "elevation": "row"}
            utils2.make_vrt(xml, data, outfolder + "/" + id + "_keep_row.vrt")
            '''newxml = xml.format(**data)
            f = open(folder + id+"_keep_row.vrt", "w")
            f.write(newxml)
            f.close()'''

            path = jsonpath + "/nearest.json"
            params = utils.json_io(path, direction="in")

            params["-txe"] = str(minx) + " " + str(maxx)
            params["-tye"] = str(miny) + " " + str(maxy)
            params["-outsize"] = str(deltx * area_multiplier) + " " + str(delty * area_multiplier)
            params["-a"]["nodata"] = str(nodata)
            params["-a"]["radius1"] = radius
            params["-a"]["radius2"] = radius
            params["src_datasource"] = outfolder + "/" + id + "_keep_row.vrt"
            params["dst_filename"] = outfolder + "/" + id + "_rows.tiff"

            # build gdal_grid request
            text = grd.build_gdal_grid_string(params, outtext=False)
            print(text)
            text = ["gdal_grid"] + text
            print(text)

            # call gdal_grid
            print("Getting the row raster")
            out, err = utils.run_tool(text)
            print("output" + out)
            if err:
                print("error:" + err)
                raise Exception("gdal grid failed")

        else: # we calculate statistics on the vector data, no need for interpolation

            return berrycolor_workflow_dense(df, id)

    # extract index from the rows
    d = gdal.Open(outfolder + "/" + id + "_rows.tiff")
    row_index, row_indexed, row_properties = gdalIO.raster_dataset_to_indexed_numpy(d, id, maxbands=1,
                                                                                    bandLocation="byrow",
                                                                                    nodata=nodata)
    print("saving indexed array to disk")
    fileIO.save_object(outfolder + "/" + id + "_rows_index", (row_index, row_indexed, row_properties))
    d = None

    # output the 4 virtual files for the 4 columns
    for clr in ["a", "b", "c", "d"]:
        data = {"layername": os.path.basename(file).split(".")[0], "fullname": file, "easting": "lon",
                "northing": "lat", "elevation": clr}
        utils2.make_vrt(xml, data, outfolder + "/" + id + "_" + clr + ".vrt")
    # output the 2 virtual files for or raw fruit count and visible fruit count
    data = {"layername": os.path.basename(file).split(".")[0], "fullname": file, "easting": "lon", "northing": "lat",
            "elevation": "raw_fruit_count"}
    utils2.make_vrt(xml, data, outfolder + "/" + id + "_rawfruit.vrt")
    data = {"layername": os.path.basename(file).split(".")[0], "fullname": file, "easting": "lon", "northing": "lat",
            "elevation": "visible_fruit_per_m"}
    utils2.make_vrt(xml, data, outfolder + "/" + id + "_visiblefruit.vrt")

    # prepare interpolation parameters
    path = jsonpath + "/invdist.json"
    params = utils.json_io(path, direction="in")
    params["-txe"] = str(minx) + " " + str(maxx)
    params["-tye"] = str(miny) + " " + str(maxy)
    params["-outsize"] = str(deltx * area_multiplier) + " " + str(delty * area_multiplier)
    params["-a"]["radius1"] = radius
    params["-a"]["radius2"] = radius
    # params["-a"]["smoothing"] = "20"
    # params["-a"]["power"] = "0"
    params["-a"]["nodata"] = str(nodata)

    # first interpolate the count data
    for clr in ["raw", "visible"]:

        params["src_datasource"] = outfolder + "/" + id + "_" + clr + "fruit.vrt"
        params["dst_filename"] = outfolder + "/" + id + "_" + clr + "fruit.tiff"

        # print(params)
        # build gdal_grid request
        text = grd.build_gdal_grid_string(params, outtext=False)
        print(text)
        text = ["gdal_grid"] + text
        print(text)

        # call gdal_grid
        print("Interpolating for count " + clr)
        out, err = utils.run_tool(text)
        print("output" + out)
        if err:
            print("error:" + err)
            raise Exception("gdal grid failed")

    # upload to numpy
    d = gdal.Open(outfolder + "/" + id + "_rawfruit.tiff")
    band = d.GetRasterBand(1)
    # apply index
    new_r_indexed_raw = gdalIO.apply_index_to_single_band(band, row_index)
    d = None

    d = gdal.Open(outfolder + "/" + id + "_visiblefruit.tiff")
    band = d.GetRasterBand(1)
    # apply index
    new_r_indexed_visible = gdalIO.apply_index_to_single_band(band, row_index)
    d = None

    # check if all pixels have a value, otherwise assign nan to nodata value (wich will not be considered for statistics)
    if new_r_indexed_raw.min() == nodata:
        warnings.warn(
            "indexed data visible berry raw counr has nodata values, the current implementation will"
            " count this pixels as nozero values", RuntimeWarning)
        new_r_indexed_raw[new_r_indexed_raw == nodata] = 'nan'  # careful nan is float
    if new_r_indexed_visible.min() == nodata:
        warnings.warn(
            "indexed data visible berry per meters has nodata values, the current implementation will"
            " count this pixels as nozero values", RuntimeWarning)
        new_r_indexed_visible[new_r_indexed_visible == nodata] = 'nan'

    stats = {}

    for clr in ["a", "b", "c", "d"]:

        params["src_datasource"] = outfolder + "/" + id + "_" + clr + ".vrt"
        params["dst_filename"] = outfolder + "/" + id + "_" + clr + ".tiff"

        # build gdal_grid request
        text = grd.build_gdal_grid_string(params, outtext=False)
        print(text)
        text = ["gdal_grid"] + text
        print(text)

        # call gdal_grid
        print("interpolating for color " + clr)
        out, err = utils.run_tool(text)
        print("output" + out)
        if err:
            print("error:" + err)
            raise Exception("gdal grid failed")

        # upload to numpy
        d = gdal.Open(outfolder + "/" + id + "_" + clr + ".tiff")
        band = d.GetRasterBand(1)
        # apply index
        new_r_indexed = gdalIO.apply_index_to_single_band(band, row_index)  # this is the index from the rows
        d = None

        # check if all pixels have a value, otherwise assign nan to nodata value
        if new_r_indexed.min() == nodata:
            warnings.warn("indexed data for colorgrade " + clr + " has nodata values, the current implementation will"
                                                                 " count this pixels as nozero values", RuntimeWarning)
            new_r_indexed[new_r_indexed == nodata] = 'nan'  # careful nan is float

        stats[clr] = {}

        for i in np.unique(row_indexed):  # get the row numbers

            area = math.pow(1 / area_multiplier, 2)  # the pixel area

            # get a mask for the current row
            mask = row_indexed == i
            # statistics for current row

            # average, std, totalarea,total nonzero area, total zeroarea, total raw fruit count,
            # average raw fruit count, std raw fruit count ,total visible fruitxm, average visible fruitXm, std visible fruitXm

            # r_indexed is 2d , while new_r_indexed and mask are 1d

            '''
            stats[clr][i] = [new_r_indexed[mask[0,:]].nanmean(), new_r_indexed[mask[0,:]].nanstd(),

                                                    #todo the sum considers nan different from 0
                                                   new_r_indexed[mask[0,:]].shape[0] * area, #could use .size?
                                                   np.count_nonzero(new_r_indexed[mask[0,:]]) * area,
                                                   new_r_indexed[mask[0,:]][new_r_indexed[mask[0,:]] == 0].shape[0] * area,

                                                   new_r_indexed_raw[mask[0,:]].nansum(),
                                                   new_r_indexed_raw[mask[0,:]].nanmean(),
                                                   new_r_indexed_raw[mask[0,:]].nanstd(),
                                                   new_r_indexed_visible[mask[0,:]].nansum(),
                                                   new_r_indexed_visible[mask[0,:]].nanmean(),
                                                   new_r_indexed_visible[mask[0,:]].nanstd()]
            '''
            stats[clr][i] = [np.nanmean(new_r_indexed[mask[0, :]]), np.nanstd(new_r_indexed[mask[0, :]]),

                             # todo the sum considers nan different from 0
                             new_r_indexed[mask[0, :]].shape[0] * area,  # could use .size?
                             np.count_nonzero(new_r_indexed[mask[0, :]]) * area,
                             new_r_indexed[mask[0, :]][new_r_indexed[mask[0, :]] == 0].shape[0] * area,

                             np.nansum(new_r_indexed_raw[mask[0, :]]),
                             np.nanmean(new_r_indexed_raw[mask[0, :]]),
                             np.nanstd(new_r_indexed_raw[mask[0, :]]),
                             np.nansum(new_r_indexed_visible[mask[0, :]]),
                             np.nanmean(new_r_indexed_visible[mask[0, :]]),
                             np.nanstd(new_r_indexed_visible[mask[0, :]])]

    return id, stats


def berrycolor_workflow_dense(df, id, row="row", cnames=["id", "row", "raw_fruit_count", "visible_fruit_per_m"], colors=["a","b","c","d"]):
    """
    This function is used to calculate the statics by row directly from tyhe original data when the data is dense 
    :param df: pandas dataframe
    :param id: the operation id
    :param new_column_names: column names like ["id", "row", "raw_fruit_count", "visible_fruit_per_m", "lat", "lon", "a", "b", "c", "d"]
    :return: an dictionary with the statistics in the form

        {"colorgrade": {row: [0.031, 0.029, 93.0, 83.75, 9.25, 118339.1, 318.11, 29.281, 213405.12, 573.66968, 61.072674],
        the list values are average, std, totalarea,total nonzero area, total zeroarea, total raw fruit count, average raw fruit count, 
        std raw fruit count ,total visible fruitxm, average visible fruitXm, std visible fruitXm
    """

    stats = {}

    # find unique row values
    rows = df[row].unique()

    for c in colors:

        stats[c] = {}

        for rowid in rows:
            stats[c][int(rowid)] = []

            # select one vineyard row
            fdf = df[df[row] == rowid]

            stats[c][int(rowid)] += [fdf[c].mean(), fdf[c].std(), np.nan, np.nan, np.nan,
                                     fdf[cnames[2]].sum(),fdf[cnames[2]].mean(),fdf[cnames[2]].std(),
                                     fdf[cnames[3]].sum(), fdf[cnames[3]].mean(),fdf[cnames[3]].std()]

    return id, stats


if __name__ == "__main__":


    ###############BERRY COLOR WORKFLOW 2################

    def test_berry_work2():
        # 1 create unique id and set the output folder
        folder = "/vagrant/code/pysdss/data/output/text/"
        id = utils.create_uniqueid()

        try:
            # 2 save the ID IN THE DATABASE
            dt.create_process(dt.default_connection, id, "claudio", dt.geotypes.interpolation.name)

            # 3 -4  dowload the pointdata from the database, create new folder #TODO: create function to read from the database
            if not os.path.exists(folder + id):
                os.mkdir(folder + id)
            outfolder = folder + id + "/"

            file = "/vagrant/code/pysdss/data/input/2016_06_17.csv"
            file = shutil.copy(file, outfolder + "/" + id + "_keep.csv")  # copy to the directory
            ##########

            # 5 set data properties: correct field names if necessary
            usecols = ["%Dataset_id", " Row", " Raw_fruit_count", " Visible_fruit_per_m", " Latitude", " Longitude",
                       "Harvestable_A", "Harvestable_B", "Harvestable_C", "Harvestable_D"]

            #this names are mandatory for the current implementation
            new_column_names = ["id", "row", "raw_fruit_count", "visible_fruit_per_m", "lat", "lon", "a", "b", "c", "d"]

            # 6 set data properties: automatically calculate the average distance between points after checking  if
            # duplicate points exists, (the user can also set an average distance from the web application)

            df = utils.remove_duplicates(file, [" Latitude", " Longitude"])
            df.to_csv(file, index=False)
            del df

            # avdist = utils.average_point_distance(file, " Longitude", " Latitude", " Row", direction="x",remove_duplicates=False)

            # 7 calculate statistics along the rows
            id, stats = berrycolor_workflow(id, file, usecols, new_column_names, outfolder, average_point_distance=None,
                                              grid=None, rowdirection="x",
                                              area_multiplier=2, filterzero=False, project="32611", rows_from_rawdata=True,
                                              nodata=-1, force_interpolation=True)
            # 8 interpret statistics
            fileIO.save_object(outfolder + "/_stats", stats)
            # stats = fileIO.load_object( folder+str(id)+"/_stats")
            interpret_result(stats, outfolder)

            # 9 save the operation state to the database

            js = json.dumps(
                {
                    "result": [
                        {
                            "type": "csvtable",
                            "name": "clorgrade_a_table",
                            "path": outfolder + "/_a_output.csv"
                        }, {
                            "type": "csvtable",
                            "name": "clorgrade_b_table",
                            "path": outfolder + "/_b_output.csv"
                        }, {
                            "type": "csvtable",
                            "name": "clorgrade_c_table",
                            "path": outfolder + "/_c_output.csv"
                        }, {
                            "type": "csvtable",
                            "name": "clorgrade_d_table",
                            "path": outfolder + "/_d_output.csv"
                        }
                    ],
                    "error": {}
                }
            )

            dt.update_process(dt.default_connection, id, "claudio", dt.geomessages.completed.name, js)

            #### use geostatistics to get the image rasterized image (indicator kriging?)

        except Exception as e:

            js = json.dumps(
                {
                    "result": [],
                    "error": {"message": str(e)}
                }
            )
            dt.update_process(dt.default_connection, id, "claudio", dt.geomessages.error.name, js)


    #test_berry_work2()

    def test_calculate_totals():
        folder = "/vagrant/code/pysdss/data/output/text/workflow2/"
        id = "aa43f20b8e0b94c9189b44fbf0c879402"

        stats = fileIO.load_object(folder + id+"/_stats")
        # using csv
        l = utils.row_length(folder + id + "/" + id + "_keep.csv", "lon", "lat", "row", conversion_factor=1)
        get_counts(stats, l, folder + id + "/")


    #test_calculate_totals()

    def test_calculate_boxes():
        folder = "/vagrant/code/pysdss/data/output/text/workflow2/aa43f20b8e0b94c9189b44fbf0c879402"
        file = folder + "/_" + "count_output.csv"
        get_boxes(file, 2000, 19)

    #test_calculate_boxes()

    def test_dense_berry_work():

        # 1 create unique id and set the output folder
        folder = "/vagrant/code/pysdss/data/output/text/"
        id = utils.create_uniqueid()

        if not os.path.exists(folder + id):
            os.mkdir(folder + id)
        outfolder = folder + id + "/"

        file = "/vagrant/code/pysdss/data/input/2016_06_17.csv"
        file = shutil.copy(file, outfolder + "/" + id + "_keep.csv")  # copy to the directory

        # 5 set data properties: correct field names if necessary
        usecols = ["%Dataset_id", " Row", " Raw_fruit_count", " Visible_fruit_per_m", " Latitude", " Longitude",
                   "Harvestable_A", "Harvestable_B", "Harvestable_C", "Harvestable_D"]

        new_column_names = ["id", "row", "raw_fruit_count", "visible_fruit_per_m", "lat", "lon", "a", "b", "c", "d"]

        # 6 set data properties: automatically calculate the average distance between points after checking  if
        # duplicate points exists, (the user can also set an average distance from the web application)

        df = utils.remove_duplicates(file, [" Latitude", " Longitude"])
        df.to_csv(file, index=False)
        del df

        # 7 calculate statistics along the rows
        id, stats = berrycolor_workflow(id, file, usecols, new_column_names, outfolder, average_point_distance=None,
                                          grid=None, rowdirection="x",
                                          area_multiplier=2, filterzero=False, project="32611", rows_from_rawdata=True,
                                          nodata=-1, force_interpolation=False) #set force_interpolation=False to skip interpolation
        # 8 interpret statistics
        fileIO.save_object(outfolder + "/_stats", stats)
        # stats = fileIO.load_object( folder+str(id)+"/_stats")
        interpret_result(stats, outfolder)

    #test_dense_berry_work()
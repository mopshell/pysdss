### tests for using krigin on colordata

### modifyng  berrycolor_workflow_1 to allow interpolation by kriging

import pandas as pd
import math
import random
import shutil
#import json
#import shutil
#import warnings


import sys
import os
import os.path
from osgeo import gdal
#from osgeo import gdalconst as gdct
#from osgeo import ogr
from osgeo import osr

import numpy as np
#from scipy.interpolate import Rbf
#from scipy.misc import imsave

import scipy.spatial.ckdtree as kdtree
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#from matplotlib import cm

from bokeh.plotting import figure
from bokeh.io import output_file, save
from bokeh.models import Label


#from rasterstats import point_query

from pysdss.utility import fileIO
from pysdss.utility import gdalRasterIO as gdalIO

from pysdss import utils
from pysdss.database import database as dt
from pysdss.filtering import filter
from pysdss.gridding import gridding as grd
from pysdss.utility import utils as utils2

import  pysdss.geostat.variance as vrc
import pysdss.geostat.fit as fit
from  pysdss.utils import rmse
from pysdss.geostat import kriging as krig


############################################   30 05 2017 ######################################

##add parameter that specify we want random filtered data for row  ( take first/last and then random in between the 2)
##add parameter to force inteerpolation for points in the same row only (for higher filters the interpolation radius will
##overlap with points belonging to adjacent rows

#so there are 8 possibilities for each interpolation method (so far inverse distance and moving average, kriging will follow)
# ordered filter, first/last, free interpolation
# ordered filter, no first/last, free interpolation
# ordered filter, first/last, interpolation by row only
# ordered filter, no first/last, interpolation by row only
# random filter, first/last, free interpolation
# random filter, no first/last, free interpolation
# random filter, first/last, interpolation by row only
# random filter, no first/last, interpolation by row only


#for random filter the radius is calculated with the max distance between points on the same row


#for the inverse distance at present the power is set to the default value 2, the interpolation should be repeated for
#different power values and then root mean square error should be calculated to decide the optimal radius
#actually it should be also possible to experiment different radius setting, therefore creating a matrix radius/power


#           pow1    pow2    pow3    ...
#   radius1  x       x       x
#   radius2  x       x       x      ...
#   radius3  x       x       x      ...
#   radius4  x       x       x      ...


def berrycolor_workflow_1(steps=[30], interp="simple", random_filter=False, first_last=False, force_interpolation_by_row=False,
                          variogram = "global",rowdirection="x", epsg="32611", id = "adca0a053062b45da8a20e0a0d4b4fe54", skip=True):
    """
    similar to interpolate_and_filter but here as input i am using the original csv file and I am considering the
    number of berries as well

    steps: the filtering steps to use
    interp: the interpolation method to use on the filtered data   "simple" or "ordinary", kriging will be added later
    random filter:   true to take random points (e,g step2 takes 50% random points for each row, false use ordered filter
    (e.g. step2 takes 1 point every 2 points)
    first_last: true to always have the first and last point of a vineyard row
    force_interpolation : true if i want to be sure the points belong to the current row only
    variogram: kind of variogram  "global","byrow","local"


                                                        variogram
                                            global      byrow       local
    force_interpolation_by_row    true      X           X           X
                                  false     X                       X

    :return:
    """



    if variogram == "local":
        raise NotImplementedError("local variogram is not implemented")

    if interp not in ["simple", "ordinary"]:
        raise ValueError("Interpolation should be 'simple' or 'ordinary'")

    if variogram not in ["global", "byrow", "local"]:
        raise ValueError("Variogram should be 'global' or 'byrow' or 'local'")

    if variogram == 'byrow' and not force_interpolation_by_row:
        raise ValueError("Variogram=='byrow' cannot be used with force_interpolation_by_row==False")

    jsonpath = os.path.join(os.path.dirname(__file__), '../..', "pysdss/experiments/interpolation/")
    jsonpath = os.path.normpath(jsonpath)

    # 1 download filtered data with the chosen and create a dataframe

    folder = "/vagrant/code/pysdss/data/output/text/"
    experimentfolder = "/vagrant/code/pysdss/experiments/interpolation/"


    #####if not skip:
    # create a folder to store the output
    id = utils.create_uniqueid()
    if not os.path.exists(folder + id):
        os.mkdir(folder + id)
    folder = folder + id + "/"

    # create a directory to store settings
    # if not os.path.exists(experimentfolder + str(id)):
    #   os.mkdir(experimentfolder + str(id))
    # dowload_data_test(id, folder)

    # dowload the data #TODO: create function to read from the database
    df = pd.read_csv("/vagrant/code/pysdss/data/input/2016_06_17.csv",
                     usecols=["%Dataset_id", " Row", " Raw_fruit_count",
                              " Visible_fruit_per_m", " Latitude", " Longitude", "Harvestable_A", "Harvestable_B",
                              "Harvestable_C", "Harvestable_D"])
    df.columns = ["id", "row", "raw_fruit_count", "visible_fruit_per_m", "lat", "lon", "a", "b", "c", "d"]

    # here should be necessary some mapping to get the column names


    # 2 filter data (and save to disk .
    # 1/03/17  we don't filter
    # keep, discard = filter.filter_byvalue(df, 0, ">", colname="d")
    keep = df

    # 3 save filtered data to disk after reprojection
    # reproject latitude, longitude #WGS 84 / UTM zone 11N
    west, north = utils2.reproject_coordinates(keep, "epsg:"+epsg, "lon", "lat", False)
    filter.set_field_value(keep, west, "lon")
    filter.set_field_value(keep, north, "lat")
    filter.set_field_value(keep, 0, "keepa")
    filter.set_field_value(keep, 0, "keepb")
    filter.set_field_value(keep, 0, "keepc")
    filter.set_field_value(keep, 0, "keepd")


    ##############search and remove duplicates
    #keep = utils.remove_duplicates(keep, ["lat", "lon"], operation="mean")
    ########################################

    keep.to_csv(folder + id + "_keep.csv", index=False)

    '''west,north =utils.reproject_coordinates(discard, "epsg:32611", "lon", "lat", False)
    filter.set_field_value(discard, west,"lon")
    filter.set_field_value(discard, north,"lat")
    filter.set_field_value(discard, 0, "keepd")
    discard.to_csv(folder + id +"_discard.csv",index=False)'''




    # 4 create the virtualdataset header for the datafile

    with open(jsonpath + "/vrtmodel.txt") as f:
        xml = f.read()

    # output the 4 virtual files for the 4 columns
    # for clr in ["a"]:
    for clr in ["a", "b", "c", "d"]:
        data = {"layername": id + "_keep", "fullname": folder + id + "_keep.csv", "easting": "lon", "northing": "lat",
                "elevation": clr}
        utils2.make_vrt(xml, data, folder + id + "_keep" + clr + ".vrt")

    # output the virtual files for raw fruit count and visible fruit count
    data = {"layername": id + "_keep", "fullname": folder + id + "_keep.csv", "easting": "lon", "northing": "lat",
            "elevation": "raw_fruit_count"}
    utils2.make_vrt(xml, data, folder + id + "_keep_rawfruit.vrt")

    data = {"layername": id + "_keep", "fullname": folder + id + "_keep.csv", "easting": "lon", "northing": "lat",
            "elevation": "visible_fruit_per_m"}
    utils2.make_vrt(xml, data, folder + id + "_keep_visiblefruit.vrt")

    # 5 define an interpolation grid (the user can choose an existing grid or draw on screen)
    # here i am doing manually for testing, i just take the bounding box

    minx = math.floor(keep['lon'].min())
    maxx = math.ceil(keep['lon'].max())
    miny = math.floor(keep['lat'].min())
    maxy = math.ceil(keep['lat'].max())
    delty = maxy - miny  # height
    deltx = maxx - minx  # width

    area_multiplier = 2  # 2 to double the resulution and halve the pixel size to 0.5 meters


    ######if not skip:

    # 6 log a running process to the database
    # 7 save grid metadata and interpolation metadata to the database historic tables
    # 8 execute gridding with gdal_array (set nodata value to zero or other value)


    path = jsonpath + "/average.json"
    params = utils.json_io(path, direction="in")

    params["-txe"] = str(minx) + " " + str(maxx)
    params["-tye"] = str(miny) + " " + str(maxy)
    # pixel is 0.5 meters
    params["-outsize"] = str(deltx * area_multiplier) + " " + str(delty * area_multiplier)

    # for clr in ["a"]:
    for clr in ["a", "b", "c", "d"]:

        params["src_datasource"] = folder + id + "_keep" + clr + ".vrt"
        params["dst_filename"] = folder + id + "_" + clr + ".tiff"

        # build gdal_grid request
        text = grd.build_gdal_grid_string(params, outtext=False)
        print(text)
        text = ["gdal_grid"] + text
        print(text)

        # call gdal_grid
        print("Getting the 'truth' raster for color " + clr)
        out, err = utils.run_tool(text)
        print("output" + out)
        if err:
            print("error:" + err)
            raise Exception("gdal grid failed")

    ##rasterize raw fruit and visible fruit

    params["src_datasource"] = folder + id + "_keep_rawfruit.vrt"
    params["dst_filename"] = folder + id + "_rawfruit.tiff"

    # build gdal_grid request
    text = grd.build_gdal_grid_string(params, outtext=False)
    print(text)
    text = ["gdal_grid"] + text
    print(text)

    # call gdal_grid
    print("Getting the 'truth' raster for raw fruit count")
    out, err = utils.run_tool(text)
    print("output" + out)
    if err:
        print("error:" + err)
        raise Exception("gdal grid failed")

    params["src_datasource"] = folder + id + "_keep_visiblefruit.vrt"
    params["dst_filename"] = folder + id + "_visiblefruit.tiff"

    # build gdal_grid request
    text = grd.build_gdal_grid_string(params, outtext=False)
    print(text)
    text = ["gdal_grid"] + text
    print(text)

    # call gdal_grid
    print("Getting the 'truth' raster for visible fruit per m")
    out, err = utils.run_tool(text)
    print("output" + out)
    if err:
        print("error:" + err)
        raise Exception("gdal grid failed")

    # 9 add result url to the database table
    # 10 log completed process (maybe should add the location of the result


    # 11 define the indexed array for the images, I used the same interpolation method and parameters therefore the index
    #  is the same

    # for clr in ["a"]:
    for clr in ["a", "b", "c", "d"]:
        d = gdal.Open(folder + id + "_" + clr + ".tiff")
        index, r_indexed, properties = gdalIO.raster_dataset_to_indexed_numpy(d, id, maxbands=1, bandLocation="byrow",
                                                                              nodata=-1)
        print("saving indexed array to disk")
        fileIO.save_object(folder + id + "_indexedarray_" + clr, (index, r_indexed, properties))
        d = None

    # define the indexed array for the raw and visible fruit
    d = gdal.Open(folder + id + "_rawfruit.tiff")
    index, r_indexed, properties = gdalIO.raster_dataset_to_indexed_numpy(d, id, maxbands=1, bandLocation="byrow",
                                                                          nodata=-1)
    print("saving indexed array to disk")
    fileIO.save_object(folder + id + "_indexedarray_rawfruit", (index, r_indexed, properties))
    d = None

    d = gdal.Open(folder + id + "_visiblefruit.tiff")
    index, r_indexed, properties = gdalIO.raster_dataset_to_indexed_numpy(d, id, maxbands=1, bandLocation="byrow",
                                                                          nodata=-1)
    print("saving indexed array to disk")
    fileIO.save_object(folder + id + "_indexedarray_visiblefruit", (index, r_indexed, properties))
    d = None

    ######## rasterize rows with nearest neighbour

    data = {"layername": id + "_keep", "fullname": folder + id + "_keep.csv", "easting": "lon", "northing": "lat",
            "elevation": "row"}
    utils2.make_vrt(xml, data, folder + id + "_keep_row.vrt")
    '''newxml = xml.format(**data)
    f = open(folder + id+"_keep_row.vrt", "w")
    f.write(newxml)
    f.close()'''

    path = jsonpath + "/nearest.json"
    params = utils.json_io(path, direction="in")

    params["-txe"] = str(minx) + " " + str(maxx)
    params["-tye"] = str(miny) + " " + str(maxy)
    params["-outsize"] = str(deltx * area_multiplier) + " " + str(delty * area_multiplier)
    params["src_datasource"] = folder + id + "_keep_row.vrt"
    params["dst_filename"] = folder + id + "_row.tiff"

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

    # upload to numpy
    d = gdal.Open(folder + id + "_row.tiff")
    band = d.GetRasterBand(1)
    # apply index
    row_indexed = gdalIO.apply_index_to_single_band(band,
                                                        index)  # this is the last index from the fruit count, all the indexes are the same
    d = None


    #################################################################################


    # collect statistics for the comparisons
    stats = {}
    # mean=[]
    # std=[]

    ########################################################
    ####folder = "/vagrant/code/pysdss/data/output/text/"
    ####folder = folder + id + "/"


    #######################################################

    ####### get the indexed array for raw count
    index_raw, r_indexed_raw, properties_raw = fileIO.load_object(folder + id + "_indexedarray_rawfruit")
    ####### get the indexed array for visible count
    index_visible, r_indexed_visible, properties_visible = fileIO.load_object(
        folder + id + "_indexedarray_visiblefruit")


    #####index = index_visible

    ###########################
    # upload to numpy
    d = gdal.Open(folder + id + "_row.tiff")
    band = d.GetRasterBand(1)
    # apply index
    row_indexed = gdalIO.apply_index_to_single_band(band,
                                                        index)  # this is the last index from the fruit count, all the indexes are the same
    d = None

    ##################################



    if force_interpolation_by_row:

        ## make temporary directory to store the interpolated vineyard rows
        #################################################
        tempfolder = folder + "/temprasters/"
        os.mkdir(tempfolder)

        # define new rasters increasing the data filtering and interpolating TODO: create configuration files for automation
        for step in steps:

            # filter data by step
            k, d = filter.filter_bystep(keep, step=step, rand=random_filter, first_last=first_last,rowname="row")
            k.to_csv(folder + id + "_keep_step" + str(step) + ".csv", index=False)
            d.to_csv(folder + id + "_discard_step" + str(step) + ".csv", index=False)

            # find unique row numbers
            rows = k["row"].unique()

            # add dictionary for this step
            stats[step] = {}
            for clr in ["a", "b", "c", "d"]:
                stats[step][clr] = {}
                for row in rows:
                    stats[step][clr][row] = {}
                    stats[step][clr][row]['truth'] = []
                    stats[step][clr][row]['interpolated'] = []

            #load xml with columns
            with open(jsonpath + "/vrtmodel_row.txt") as f:
                xml_row = f.read()

            # for clr in ["a"]:
            for clr in ["a", "b", "c", "d"]:
                data = {"layername": id + "_keep_step" + str(step),
                        "fullname": folder + id + "_keep_step" + str(step) + ".csv", "easting": "lon",
                        "northing": "lat",
                        "elevation": clr, "rowname": "row"}
                utils2.make_vrt(xml_row, data, folder + id + "_keep" + clr + "_step" + str(step) + ".vrt")


            data = {"layername": id + "_keep_step" + str(step),
                    "fullname": folder + id + "_keep_step" + str(step) + ".csv", "easting": "lon", "northing": "lat",
                    "elevation": "raw_fruit_count", "rowname": "row"}
            utils2.make_vrt(xml_row, data, folder + id + "_keep_raw_step" + str(step) + ".vrt")


            data = {"layername": id + "_keep_step" + str(step),
                    "fullname": folder + id + "_keep_step" + str(step) + ".csv", "easting": "lon", "northing": "lat",
                    "elevation": "visible_fruit_per_m", "rowname": "row"}
            utils2.make_vrt(xml_row, data, folder + id + "_keep_visible_step" + str(step) + ".vrt")



            # prepare interpolation parameters
            if interp == "invdist":
                path = jsonpath + "/invdist.json"
            if interp == "average":
                path = jsonpath + "/average.json"

            params = utils.json_io(path, direction="in")

            params["-txe"] = str(minx) + " " + str(maxx)
            params["-tye"] = str(miny) + " " + str(maxy)
            params["-outsize"] = str(deltx * area_multiplier) + " " + str(delty * area_multiplier)


            for clr in ["raw", "visible"]:

                #for each row if random calculate the max distance for the current row
                #execute gdalgrid for jus the row points
                #calculate statistics for the row and fill in the output data structure


                for row in rows:
                    # select one vineyard row
                    rowdf = k[k["row"] == int(row)]

                    # if random_filter we will set the radius to the max distance between points in a row
                    if random_filter: max_point_distance = utils.max_point_distance(rowdf, "lon", "lat", "row",
                                                                                    direction=rowdirection,
                                                                                    rem_duplicates=False)

                    if random_filter:  # if we have random points we set we set the radius to the max distance between points plus a buffer of 0.5
                        params["-a"]["radius1"] = str(max_point_distance + 0.5)
                        params["-a"]["radius2"] = str(max_point_distance + 0.5)
                    else:
                        params["-a"]["radius1"] = str(0.5 * step)
                        params["-a"]["radius2"] = str(0.5 * step)

                    #this will interpolate only the current row data
                    params["-where"] = "row="+ str(int(row))

                    params["src_datasource"] = folder + id + "_keep_" + clr + "_step" + str(step) + ".vrt"
                    params["dst_filename"] = tempfolder + id + "_" + clr + "_step" + str(step) + "_row"+str(row)+".tiff"

                    print(params)

                    # build gdal_grid request
                    text = grd.build_gdal_grid_string(params, outtext=False)
                    print(text)
                    text = ["gdal_grid"] + text
                    print(text)

                    # call gdal_grid
                    print("Getting filtered raster by step " + str(step) + "for field " + clr +" and row" + str(row))
                    out, err = utils.run_tool(text)
                    print("output" + out)
                    if err:
                        print("error:" + err)
                        raise Exception("gdal grid failed")


            # extract statistics for raw and visible
            for clr in ["a", "b", "c", "d"]:
                for row in rows:
                    # upload to numpy
                    d = gdal.Open(tempfolder + id + "_raw_step" + str(step) + "_row"+str(row) + ".tiff")
                    band = d.GetRasterBand(1)
                    # apply index
                    new_r_indexed_raw = gdalIO.apply_index_to_single_band(band,
                                                                          index)  # this is the last index from the fruit count
                    d = None

                    d = gdal.Open(tempfolder + id + "_visible_step" + str(step) + "_row"+str(row) + ".tiff")
                    band = d.GetRasterBand(1)
                    # apply index
                    new_r_indexed_visible = gdalIO.apply_index_to_single_band(band,
                                                                              index)  # this is the last index from the fruit count
                    d = None

                    #update structure with partial data
                    # get a mask for the current row
                    mask = row_indexed == row
                    # statistics for current row
                    # average, std, totalarea,total nonzero area, total zeroarea, total raw fruit count,
                    # average raw fruit count, std raw fruit count ,total visible fruitxm, average visible fruitXm, std visible fruitXm

                    stats[step][clr][row]['truth'] = [None, None, None, None, None,
                                          r_indexed_raw[0, :][mask].sum(),
                                          r_indexed_raw[0, :][mask].mean(),
                                          r_indexed_raw[0, :][mask].std(),
                                          r_indexed_visible[0, :][mask].sum(),
                                          r_indexed_visible[0, :][mask].mean(),
                                          r_indexed_visible[0, :][mask].std()]

                    stats[step][clr][row]['interpolated'] = [None, None, None, None, None,
                                                 new_r_indexed_raw[mask].sum(),
                                                 new_r_indexed_raw[mask].mean(),
                                                 new_r_indexed_raw[mask].std(),
                                                 new_r_indexed_visible[mask].sum(),
                                                 new_r_indexed_visible[mask].mean(),
                                                 new_r_indexed_visible[mask].std()]

            for clr in ["a", "b", "c", "d"]:

                #for each row if random calculate the max distance for the current row
                #execute gdalgrid for jus the row points
                #calculate statistics for the row and fill in the output data structure

                for row in rows:
                    # select one vineyard row
                    rowdf = k[k["row"] == int(row)]

                    # if random_filter we will set the radius to the max distance between points in a row
                    if random_filter: max_point_distance = utils.max_point_distance(rowdf, "lon", "lat", "row",
                                                                                    direction=rowdirection,
                                                                                    rem_duplicates=False)

                    if random_filter:  # if we have random points we set we set the radius to the max distance between points plus a buffer of 0.5
                        params["-a"]["radius1"] = str(max_point_distance + 0.5)
                        params["-a"]["radius2"] = str(max_point_distance + 0.5)
                    else:
                        params["-a"]["radius1"] = str(0.5 * step)
                        params["-a"]["radius2"] = str(0.5 * step)

                    #this will interpolate only the current row data
                    params["-where"] = "row="+ str(int(row))

                    params["src_datasource"] = folder + id + "_keep" + clr + "_step" + str(step) + ".vrt"
                    params["dst_filename"] = tempfolder + id + "_" + clr + "_step" + str(step) + "_row"+str(row) + ".tiff"

                    # build gdal_grid request
                    text = grd.build_gdal_grid_string(params, outtext=False)
                    print(text)
                    text = ["gdal_grid"] + text
                    print(text)

                    # call gdal_grid
                    print("Getting filtered raster by step " + str(step) + "for color grade " + clr)
                    out, err = utils.run_tool(text)
                    print("output" + out)
                    if err:
                        print("error:" + err)
                        raise Exception("gdal grid failed")

                    # upload to numpy
                    d = gdal.Open(tempfolder + id + "_" + clr + "_step" + str(step) + "_row"+str(row) + ".tiff")
                    band = d.GetRasterBand(1)
                    # apply index
                    new_r_indexed = gdalIO.apply_index_to_single_band(band,
                                                                      index)  # this is the last index from the fruit count
                    d = None

                    # calculate average, std, total area (here a pixel is 0.25 m2,
                    # totla area for pixel > 0, total area for pixels with 0 value

                    ####### get the indexed array for this colour grade

                    # d = gdal.Open(folder + id + "_" + clr + ".tiff")
                    # band = d.GetRasterBand(1)
                    # r_indexed = gdalIO.apply_index_to_single_band(band, index) # this is the last index from the fruit count
                    # d = None
                    index, r_indexed, properties = fileIO.load_object(folder + id + "_indexedarray_" + clr)

                    # update structure with partial data
                    area = math.pow(1 / area_multiplier, 2)
                    # get a mask for the current row
                    mask = row_indexed == row
                    #fill in with data
                    partial_truth_data = stats[step][clr][row]['truth']
                    partial_truth_data[:5] = [r_indexed[0, :][mask].mean(), r_indexed[0, :][mask].std(),
                    r_indexed[0, :][mask].shape[0] * area,
                    np.count_nonzero(r_indexed[0, :][mask]) * area,
                    r_indexed[0, :][mask][r_indexed[0, :][mask] == 0].shape[0] * area]

                    partial_interpolated_data = stats[step][clr][row]['interpolated']
                    partial_interpolated_data[:5] = [new_r_indexed[mask].mean(), new_r_indexed[mask].std(),
                    new_r_indexed[mask].shape[0] * area,
                    np.count_nonzero(new_r_indexed[mask]) * area,
                    new_r_indexed[mask][new_r_indexed[mask] == 0].shape[0] * area]



    else: # no force interpolation by row

        # define new rasters increasing the data filtering and interpolating
        for step in steps:
            # add dictionary for this step
            stats[step] = {}

            # filter data by step
            k, d = filter.filter_bystep(keep, step=step, rand=random_filter, first_last=first_last,rowname="row")
            # if random_filter we will set the radius to the max distance between points in a row
            if random_filter: max_point_distance = utils.max_point_distance(k, "lon", "lat", "row", direction=rowdirection)

            '''
            i already projected before
            west, north = utils.reproject_coordinates(k, "epsg:32611", "lon", "lat", False)
            filter.set_field_value(k, west, "lon")
            filter.set_field_value(k, north, "lat")
            filter.set_field_value(k, 0, "keepa")
            filter.set_field_value(k, 0, "keepb")
            filter.set_field_value(k, 0, "keepc")
            filter.set_field_value(k, 0, "keepd")'''
            k.to_csv(folder + id + "_keep_step" + str(step) + ".csv", index=False)

            '''west, north = utils.reproject_coordinates(d, "epsg:32611", "lon", "lat", False)
            filter.set_field_value(d, west, "lon")
            filter.set_field_value(d, north, "lat")
            filter.set_field_value(d, 0, "keepa")
            filter.set_field_value(d, 0, "keepb")
            filter.set_field_value(d, 0, "keepc")
            filter.set_field_value(d, 0, "keepd")'''
            d.to_csv(folder + id + "_discard_step" + str(step) + ".csv", index=False)

            if random_filter: radius = max_point_distance + 0.5
            else: radius = 0.5 * step

            ''' grid parameters to build the numpy array
            str(minx) + " " + str(maxx)
            str(miny) + " " + str(maxy)
            str(deltx * area_multiplier) + " " + str(delty * area_multiplier)
            '''

            #extract the data to be used to build the variogram
            vardata = k[["lon", "lat", "visible_fruit_per_m"]].values
            varmean = np.mean( vardata[:, 2])

            # here we calculate the global variogram for visible_fruit_per_m
            if variogram == "global":


                ####if not skip:

                print("calculte global empirical semivariance")

                #i set the hh to 5 neters and the max distance to the smallest field edge
                hh = 10
                #lags = range(0,min(delty ,deltx), hh)
                lags = range(0, int((delty+deltx)/2), hh)

                gamma = vrc.semivar(vardata, lags, hh)
                # covariance = vrc.covar(data, lags, hh)


                # chart empirical semivariance and covariance
                output_file(folder + id + "_semivariogram_step"+str(step)+".html")
                f = figure()
                f.line(gamma[0, :], gamma[1, :], line_color="green", line_width=2, legend="Semivariance")
                f.square(gamma[0, :], gamma[1, :], fill_color=None, line_color="green", legend="Semivariance")
                #f.line(cov[0, :], cov[1, :], line_color="red", line_width=2, legend="Covariance")
                #f.square(cov[0, :], cov[1, :], fill_color=None, line_color="red", legend="Covariance")
                f.legend.location = "top_left"
                save(f)


                print("fit semivariogram for visible_fruit_per_m")
                # choose the model with lowest rmse
                semivariograms = []
                semivariograms.append(fit.fitsemivariogram(vardata, gamma, fit.spherical))
                semivariograms.append(fit.fitsemivariogram(vardata, gamma, fit.linear))
                semivariograms.append(fit.fitsemivariogram(vardata, gamma, fit.gaussian))
                semivariograms.append(fit.fitsemivariogram(vardata, gamma, fit.exponential))
                rsmes = [rmse(gamma[0], i(gamma[0])) for i in semivariograms]
                semivariogram = semivariograms[rsmes.index(min(rsmes))]

                # chart the fitted models
                output_file(folder + id + "_semivariogram_step"+str(step)+"_fitted.html")
                f = figure()
                f.circle(gamma[0], gamma[1], fill_color=None, line_color="blue")
                f.line(gamma[0], semivariogram(gamma[0]), line_color="red", line_width=2)
                f.legend.location = "top_left"
                save(f)

                fileIO.save_object(folder + id + "_semivariogram_step"+str(step)+".csv",semivariogram, usedill=True)

                ####sys.exit()

                ####else:
                ####    semivariogram = fileIO.load_object(folder + id + "_semivariogram_step"+str(step)+".csv", usedill=True)

            #initialize the global kdtree index

            xydata = k[["lon","lat"]].values
            tree = kdtree.cKDTree(xydata)


            ###########interpolation for visible berries per meter##############

            #initialize grid as numpy array, filled in with nodata value (-1)

            grid = np.ones((delty * area_multiplier, deltx * area_multiplier))*(-1)
            grid_error = np.ones((delty * area_multiplier, deltx * area_multiplier))*(-1)

            # initialize a geotransform
            geotr = (minx, 1/area_multiplier, 0, maxy, 0, -1/area_multiplier)

            '''
            lon, lat = ApplyGeoTransform(gt,x,y)
            inv_gt = gdal.InvGeoTransform(in_gt)
            if gdal.VersionInfo()[0] == '1':
                if inv_gt[0] == 1:
                    inv_gt = inv_gt[1]
            else:
                raise RuntimeError('Inverse geotransform failed')
            elif inv_gt is None:
                raise RuntimeError('Inverse geotransform failed')
            '''

            #here I may apply the index to reduce the number of cells to iterate
            #iterate grid cells
            #index, r_indexed_visible, properties_visible = fileIO.load_object(folder + id + "_indexedarray_visiblefruit")

            print("kriging!")

            #get the number of pixels to iterate
            npixels = index[0].shape[0]

            #iterate the pixels from the index
            for i in range(npixels):
                # index[0] is for the rows, index[1] is for the columns
                lonlat = gdal.ApplyGeoTransform(geotr,int(index[1][i]),int(index[0][i]))

                ###get the  nearest cells inside the interpolation radius
                ###nearindexes = tree.query_ball_point(lonlat,radius, n_jobs= -1)
                ###if not nearindexes: #actually this shouldnt happen
                ###    continue
                ###nearvardata = vardata[nearindexes]


                #kriging with the nearest 20 points
                distance, idx = tree.query((lonlat[0], lonlat[1]), 10)
                nearvardata = vardata[idx]



                #if there is only 1 neighbour we take its value else we do kriging
                #if len(nearindexes) == 1:
                if len(idx) == 1:
                    grid[index[0][i],index[1][i]] = nearvardata[0,2] #i did this otherwise kriging with 1 point will raise error
                    grid_error[index[0][i],index[1][i]] = 0  #todo how to deal with the error?
                else:

                    #dists = pdist(xydata[nearindexes])  #
                    #####distance, idx = tree.query((lonlat[0],lonlat[1]), len(nearindexes))


                    krigdata = np.hstack((nearvardata, np.expand_dims(distance, axis=0).T))


                    if variogram == "local":
                        #"local calculate best variogram and kringing"
                        pass


                    else:
                        #else kriging with global variogram
                        if interp == "simple":
                            try:
                                simple = krig.simple(krigdata, varmean, semivariogram)
                            except Exception as e:
                                print(".", end='') #TODO did this because of the singular matrix error
                            else:
                                grid[index[0][i],index[1][i]]  = simple[0]
                                grid_error[index[0][i],index[1][i]] = simple[1]

                        else:  # ordinary
                            try:
                                ordinary = krig.ordinary(krigdata, semivariogram)
                            except Exception as e:
                                print(".",end='') #TODO did this because of the singular matrix error
                            else:
                                grid[index[0][i],index[1][i]] = ordinary[0]
                                grid_error[index[0][i],index[1][i]] = ordinary[1]

            # save to  folder + id + "_visible_step" + str(step) +  ".tiff"

            outRaster = None
            try:
                rastergrid = folder + id + "_visible_step" + str(step) + "_" + interp + variogram + ".tiff"
                driver = gdal.GetDriverByName('GTiff')
                outRaster = driver.Create(rastergrid, deltx * area_multiplier,delty * area_multiplier, 1, gdal.GDT_Float32)
                outRaster.SetGeoTransform(geotr)
                outband = outRaster.GetRasterBand(1)
                outband.SetNoDataValue(-1.0)
                outband.WriteArray( np.flip(grid,0))
                outRasterSRS = osr.SpatialReference()
                outRasterSRS.ImportFromEPSG(int(epsg))
                outRaster.SetProjection(outRasterSRS.ExportToWkt())
                outband.FlushCache()
            finally:
                if outRaster: outRaster=None
            try:
                rastergriderror = folder + id + "_visible_step" + str(step) + "_" + interp + variogram + "_error.tiff"
                driver = gdal.GetDriverByName('GTiff')
                outRaster = driver.Create(rastergriderror, deltx * area_multiplier,delty * area_multiplier, 1, gdal.GDT_Float32)
                outRaster.SetGeoTransform(geotr)
                outband = outRaster.GetRasterBand(1)
                outband.SetNoDataValue(-1.0)
                outband.WriteArray( np.flip(grid_error,0))
                outRasterSRS = osr.SpatialReference()
                outRasterSRS.ImportFromEPSG(int(epsg))
                outRaster.SetProjection(outRasterSRS.ExportToWkt())
                outband.FlushCache()
            finally:
                if outRaster: outRaster=None


            '''
            # upload to numpy
            d = gdal.Open(folder + id + "_raw_step" + str(step) + ".tiff")
            band = d.GetRasterBand(1)
            # apply index
            new_r_indexed_raw = gdalIO.apply_index_to_single_band(band,
                                                                  index)  # this is the last index from the fruit count
            d = None
            '''


            d = gdal.Open(folder + id + "_visible_step" + str(step) + "_" + interp + variogram + ".tiff")
            band = d.GetRasterBand(1)
            # apply index
            new_r_indexed_visible = gdalIO.apply_index_to_single_band(band,
                                                                      index)  # this is the last index from the fruit count
            d = None


            # for clr in ["a"]:
            for clr in ["a", "b", "c", "d"]:

                #dst_filename = folder + id + "_" + clr + "_step" + str(step) + ".tiff"
                #if variogram == "global":pass

                # initialize grid as numpy array, filled in with nodata value (-1)

                grid = np.ones((delty * area_multiplier, deltx * area_multiplier)) * (-1)
                grid_error = np.ones((delty * area_multiplier, deltx * area_multiplier)) * (-1)

                # initialize a geotransform
                geotr = (minx, 1 / area_multiplier, 0, maxy, 0, -1 / area_multiplier)


                # extract the data to be used to build the variogram
                vardata = k[["lon", "lat", clr]].values
                varmean = np.mean(vardata[:, 2])

                # here we calculate the global variogram for colorgrade
                if variogram == "global":

                    ####if not skip:

                    print("calculte global empirical semivariance")

                    # i set the hh to 5 neters and the max distance to the smallest field edge
                    hh = 10
                    # lags = range(0,min(delty ,deltx), hh)
                    lags = range(0, int((delty + deltx) / 2), hh)

                    gamma = vrc.semivar(vardata, lags, hh)
                    # covariance = vrc.covar(data, lags, hh)


                    # chart empirical semivariance and covariance
                    output_file(folder + id + "_semivariogram_step" + str(step) + ".html")
                    f = figure()
                    f.line(gamma[0, :], gamma[1, :], line_color="green", line_width=2, legend="Semivariance")
                    f.square(gamma[0, :], gamma[1, :], fill_color=None, line_color="green", legend="Semivariance")
                    # f.line(cov[0, :], cov[1, :], line_color="red", line_width=2, legend="Covariance")
                    # f.square(cov[0, :], cov[1, :], fill_color=None, line_color="red", legend="Covariance")
                    f.legend.location = "top_left"
                    save(f)

                    print("fit semivariogram for color "+ clr)
                    # choose the model with lowest rmse
                    semivariograms = []
                    semivariograms.append(fit.fitsemivariogram(vardata, gamma, fit.spherical))
                    semivariograms.append(fit.fitsemivariogram(vardata, gamma, fit.linear))
                    semivariograms.append(fit.fitsemivariogram(vardata, gamma, fit.gaussian))
                    semivariograms.append(fit.fitsemivariogram(vardata, gamma, fit.exponential))
                    rsmes = [rmse(gamma[0], i(gamma[0])) for i in semivariograms]
                    semivariogram = semivariograms[rsmes.index(min(rsmes))]

                    # chart the fitted models
                    output_file(folder + id + "_semivariogram_step" + str(step) + "_" + clr + "_" + "_fitted.html")
                    f = figure()
                    f.circle(gamma[0], gamma[1], fill_color=None, line_color="blue")
                    f.line(gamma[0], semivariogram(gamma[0]), line_color="red", line_width=2)
                    f.legend.location = "top_left"
                    save(f)

                    fileIO.save_object(folder + id + "_semivariogram_step" + str(step) + "_" + clr, semivariogram,
                                       usedill=True)

                    #####sys.exit()

                    ####else:
                    ####    semivariogram = fileIO.load_object(folder + id + "_semivariogram_step" + str(step) + "_" + clr,
                    ####                                       usedill=True)

                print("kriging color "+clr+" step "+str(step))

                # get the number of pixels to iterate
                npixels = index[0].shape[0]

                # iterate the pixels from the index
                for i in range(npixels):
                    # index[0] is for the rows, index[1] is for the columns
                    lonlat = gdal.ApplyGeoTransform(geotr, int(index[1][i]), int(index[0][i]))

                    ###get the  nearest cells inside the interpolation radius
                    ###nearindexes = tree.query_ball_point(lonlat,radius, n_jobs= -1)
                    ###if not nearindexes: #actually this shouldnt happen
                    ###    continue
                    ###nearvardata = vardata[nearindexes]


                    # kriging with the nearest 20 points
                    distance, idx = tree.query((lonlat[0], lonlat[1]), 10)
                    nearvardata = vardata[idx]

                    # if there is only 1 neighbour we take its value else we do kriging
                    # if len(nearindexes) == 1:
                    if len(idx) == 1:
                        grid[index[0][i], index[1][i]] = nearvardata[0, 2]  # i did this otherwise kriging with 1 point will raise error
                        grid_error[index[0][i], index[1][i]] = 0  # todo how to deal with the error?
                    else:

                        # dists = pdist(xydata[nearindexes])  #
                        ###distance, idx = tree.query((lonlat[0], lonlat[1]), len(nearindexes))

                        krigdata = np.hstack((nearvardata, np.expand_dims(distance, axis=0).T))

                        if variogram == "local":
                            # "local calculate best variogram and kringing"
                            pass


                        else:
                            # else kriging with global variogram
                            if interp == "simple":
                                try:
                                    simple = krig.simple(krigdata, varmean, semivariogram)
                                except Exception as e:
                                    print(".", end='')  # TODO did this because of the singular matrix error
                                else:
                                    grid[index[0][i], index[1][i]] = simple[0]
                                    grid_error[index[0][i], index[1][i]] = simple[1]
                            else:  # ordinary
                                try:
                                    ordinary = krig.ordinary(krigdata, semivariogram)
                                except Exception as e:
                                    print(".", end='')  # TODO did this because of the singular matrix error
                                else:
                                    grid[index[0][i], index[1][i]] = ordinary[0]
                                    grid_error[index[0][i], index[1][i]] = ordinary[1]

                # save to folder + id + "_" + clr + "_step" + str(step) + "_" + interp + variogram + ".tiff")

                outRaster = None
                try:
                    rastergrid = folder + id + "_"+clr+"_step" + str(step) + "_" + interp + variogram + ".tiff"
                    driver = gdal.GetDriverByName('GTiff')
                    outRaster = driver.Create(rastergrid, deltx * area_multiplier, delty * area_multiplier, 1,
                                              gdal.GDT_Float32)
                    outRaster.SetGeoTransform(geotr)
                    outband = outRaster.GetRasterBand(1)
                    outband.SetNoDataValue(-1.0)
                    outband.WriteArray(np.flip(grid, 0))
                    outRasterSRS = osr.SpatialReference()
                    outRasterSRS.ImportFromEPSG(int(epsg))
                    outRaster.SetProjection(outRasterSRS.ExportToWkt())
                    outband.FlushCache()
                finally:
                    if outRaster: outRaster = None

                outRaster = None
                try:
                    rastergriderror = folder + id + "_"+clr+"_step" + str(step) + "_" + interp + variogram + "_error.tiff"
                    driver = gdal.GetDriverByName('GTiff')
                    outRaster = driver.Create(rastergriderror, deltx * area_multiplier, delty * area_multiplier, 1,
                                              gdal.GDT_Float32)
                    outRaster.SetGeoTransform(geotr)
                    outband = outRaster.GetRasterBand(1)
                    outband.SetNoDataValue(-1.0)
                    outband.WriteArray(np.flip(grid_error, 0))
                    outRasterSRS = osr.SpatialReference()
                    outRasterSRS.ImportFromEPSG(int(epsg))
                    outRaster.SetProjection(outRasterSRS.ExportToWkt())
                    outband.FlushCache()
                finally:
                    if outRaster: outRaster = None


                # upload to numpy
                d = gdal.Open(folder + id + "_"+clr+"_step" + str(step) + "_" + interp + variogram + ".tiff")
                band = d.GetRasterBand(1)
                # apply index
                new_r_indexed = gdalIO.apply_index_to_single_band(band,
                                                                  index)  # this is the last index from the fruit count
                d = None

                # calculate average, std, total area (here a pixel is 0.25 m2,
                # total area for pixel > 0, total area for pixels with 0 value

                ####### get the indexed array for this colour grade

                # d = gdal.Open(folder + id + "_" + clr + ".tiff")
                # band = d.GetRasterBand(1)
                # r_indexed = gdalIO.apply_index_to_single_band(band, index) # this is the last index from the fruit count
                # d = None
                index, r_indexed, properties = fileIO.load_object(folder + id + "_indexedarray_" + clr)

                stats[step][clr] = {}

                for i in np.unique(row_indexed):  # get the row numbers

                    area = math.pow(1 / area_multiplier, 2)

                    # add dictionary for this row
                    stats[step][clr][i] = {}

                    # get a mask for the current row
                    mask = row_indexed == i
                    # statistics for current row

                    # average, std, totalarea,total nonzero area, total zeroarea, total raw fruit count,
                    # average raw fruit count, std raw fruit count ,total visible fruitxm, average visible fruitXm, std visible fruitXm

                    # r_indexed is 2d , while new_r_indexed and mask are 1d
                    stats[step][clr][i]['truth'] = [r_indexed[0, :][mask].mean(), r_indexed[0, :][mask].std(),
                                                    r_indexed[0, :][mask].shape[0] * area,
                                                    np.count_nonzero(r_indexed[0, :][mask]) * area,
                                                    r_indexed[0, :][mask][r_indexed[0, :][mask] == 0].shape[0] * area,
                                                    #r_indexed_raw[0, :][mask].sum(),
                                                    #r_indexed_raw[0, :][mask].mean(),
                                                    #r_indexed_raw[0, :][mask].std(),
                                                    r_indexed_visible[0, :][mask].sum(),
                                                    r_indexed_visible[0, :][mask].mean(),
                                                    r_indexed_visible[0, :][mask].std()]

                    '''
                    stats[step][clr][i]['truth'] = [r_indexed[mask].mean(), r_indexed[mask].std(),
                                                    r_indexed[mask].shape[0]*area, np.count_nonzero(r_indexed[mask])*area,
                                                    r_indexed[mask][r_indexed[mask] == 0].shape[0]*area,
                                                    r_indexed_raw[0,:][mask].sum(), r_indexed_raw[0,:][mask].mean(),r_indexed_raw[0,:][mask].std(),
                                                    r_indexed_visible[0,:][mask].sum(),r_indexed_visible[0,:][mask].mean()], r_indexed_visible[0,:][mask].std()]'''

                    stats[step][clr][i]['interpolated'] = [new_r_indexed[mask].mean(), new_r_indexed[mask].std(),
                                                           new_r_indexed[mask].shape[0] * area,
                                                           np.count_nonzero(new_r_indexed[mask]) * area,
                                                           new_r_indexed[mask][new_r_indexed[mask] == 0].shape[0] * area,
                                                           #new_r_indexed_raw[mask].sum(),
                                                           #new_r_indexed_raw[mask].mean(),
                                                           #new_r_indexed_raw[mask].std(),
                                                           new_r_indexed_visible[mask].sum(),
                                                           new_r_indexed_visible[mask].mean(),
                                                           new_r_indexed_visible[mask].std()]

                    # compare = r_indexed[0,:][mask] - new_r_indexed[mask]
                    # stats[clr][step][i]['comparison'] = [compare.mean(),compare.std()]
                    # pass
                    # calculate average and variance for this surface

                    # mean.append(compare.mean())
                    # std.append(compare.std())


    # print(mean)
    # print(std)
    return id, stats
    # {2: {"a": {1: {'truth': {}, 'interpolated': {}}}}}


if __name__ == "__main__":


    id, stats = berrycolor_workflow_1(steps=[10,20,30,50], interp="ordinary", random_filter=False, first_last=False, force_interpolation_by_row=False,
                          variogram = "global",rowdirection="x", epsg="32611")

    folder = "/vagrant/code/pysdss/data/output/text/" + id + "/"
    fileIO.save_object(folder + id + "_statistics_ordinary", stats)


    from pysdss.analytics.colorgrade.research.colorgrade_comparisons import test_interpret_result
    from pysdss.analytics.colorgrade.research.colorgrade_comparisons import test_compare_raster_totruth

    #id = "adca0a053062b45da8a20e0a0d4b4fe54"
    folder = "/vagrant/code/pysdss/data/output/text/" + id + "/"

    colmodel = ["avg", "std", "area", "nozero_area", "zero_area", "visible_count",
                "visible_avg", "visible_std"]

    stats = fileIO.load_object(folder + id + "_statistics_ordinary")
    test_interpret_result(stats,id,folder,colmodel)


    #point = folder + id + "_keep.csv"
    #test_compare_raster_totruth(point,"a0857fd6fbb0e4f6fbacb26b1a984779a", folder,step=steps )
    #test_compare_raster_totruth(point, id, folder,step=steps, clean=[[-1],[0]])
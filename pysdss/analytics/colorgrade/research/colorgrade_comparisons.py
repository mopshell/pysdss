# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        colorgrade_comparisond.py
# Purpose:     code for colorgrade analytics research
#               comparison between interpolation of dense data with filtered data with comparison statistics
#               this is just for research purpose and not for the final library
# Author:      claudio piccinini
#
# Created:     04/04/2017
#-------------------------------------------------------------------------------



import pandas as pd
import math
import random
#import json
#import shutil
#import warnings

import os
import os.path
from osgeo import gdal
#from osgeo import gdalconst as gdct
#from osgeo import ogr
from osgeo import osr

import numpy as np
#from scipy.interpolate import Rbf
#from scipy.misc import imsave

import matplotlib
matplotlib.use('Agg')
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

#######################################################
##############old berry workflow 1#####################

connstring = {"dbname": "grapes", "user": "claudio", "password": "claudio", "port": "5432", "host": "127.0.0.1"}
datasetid = 26
boundid = 1


########## download generic 2 joined data table
def dowload_all_data_test(id, path):
    """
    download data for all 4 grades
    :param id:
    :param path:
    :return:
    """
    leftclm = ["row", "id_sdata", "lat", "lon"]
    rightclmn = ["a", "b", "c", "d", "keepa::int", "keepb::int", "keepc::int", "keepd::int"]

    outfile = path + id + ".csv"
    # complete dataset
    dt.download_data("id_sdata", "data.SensoDatum", "id_sensor", leftclm, "id_sdata", "data.colorgrade", rightclmn,
                     datasetid, outfile, orderclms=["row", "id_sdata"], conndict=connstring)
    name = utils.create_uniqueid()
    outfile = "/vagrant/code/pysdss/data/output/" + name + ".csv"
    # clipped dataset, result ordered
    dt.download_data("id_sdata", "data.SensoDatum", "id_sensor", leftclm, "id_sdata", "data.colorgrade", rightclmn,
                     datasetid, outfile, leftboundgeom="geom", boundtable="data.boundary", righttboundgeom="geom",
                     boundid="id_bound",
                     boundvalue=1, orderclms=["row", "id_sdata"], conndict=connstring)

def test_interpolate_and_filter():
    """

    :return:
    """

    # 1 download filtered data with the chosen and create a dataframe

    folder = "/vagrant/code/pysdss/data/output/text/"
    experimentfolder = "/vagrant/code/pysdss/experiments/interpolation/"

    id = utils.create_uniqueid()

    # create a directory to store settings
    # if not os.path.exists(experimentfolder + str(id)):
    #   os.mkdir(experimentfolder + str(id))


    # dowload_data_test(id, folder)
    dowload_all_data_test(id, folder)

    df = pd.read_csv(folder + id + ".csv")
    # strip and lower column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lower()

    # here should be necessary some mapping to get the column names


    # 2 filter data and save to disk . this is the "truth" so we just delete the zeros
    # 1/03/17  we don't filter
    # keep, discard = filter.filter_byvalue(df, 0, ">", colname="d")
    keep = df

    # 3 save filtered data to disk after reprojection
    # reproject latitude, longitude #WGS 84 / UTM zone 11N
    west, north = utils2.reproject_coordinates(keep, "epsg:32611", "lon", "lat", False)
    filter.set_field_value(keep, west, "lon")
    filter.set_field_value(keep, north, "lat")
    filter.set_field_value(keep, 0, "keepa")
    filter.set_field_value(keep, 0, "keepb")
    filter.set_field_value(keep, 0, "keepc")
    filter.set_field_value(keep, 0, "keepd")
    keep.to_csv(folder + id + "_keep.csv", index=False)

    '''west,north =utils.reproject_coordinates(discard, "epsg:32611", "lon", "lat", False)
    filter.set_field_value(discard, west,"lon")
    filter.set_field_value(discard, north,"lat")
    filter.set_field_value(discard, 0, "keepd")
    discard.to_csv(folder + id +"_discard.csv",index=False)'''

    # 4 create the virtualdataset header for the datafile

    xml = '''<OGRVRTDataSource>
                <OGRVRTLayer name="{layername}">
                    <SrcDataSource>{fullname}</SrcDataSource>
                    <GeometryType>wkbPoint</GeometryType>
                    <GeometryField encoding="PointFromColumns" x="{easting}" y="{northing}" z="{elevation}"/>
                </OGRVRTLayer>
    </OGRVRTDataSource>
    '''

    # output the 4 virtual files for the 4 columns
    # for clr in ["a"]:
    for clr in ["a", "b", "c", "d"]:
        data = {"layername": id + "_keep", "fullname": folder + id + "_keep.csv", "easting": "lon", "northing": "lat",
                "elevation": clr}
        newxml = xml.format(**data)
        f = open(folder + id + "_keep" + clr + ".vrt", "w")
        f.write(newxml)
        f.close()

    # 5 define an interpolation grid (the user can choose an existing grid or draw on screen)
    # here i am doing manually for testing, i just take the bounding box

    minx = math.floor(keep['lon'].min())
    maxx = math.ceil(keep['lon'].max())
    miny = math.floor(keep['lat'].min())
    maxy = math.ceil(keep['lat'].max())
    delty = maxy - miny
    deltx = maxx - minx

    # 6 log a running process to the database
    # 7 save grid metadata and interpolation metadata to the database historic tables
    # 8 execute gridding with gdal_array (set nodata value to zero or other value)


    algo = {
        "name": "average",
        "radius1": "0.4",
        "radius2": "0.4",
        "angle": "0",
        "min_points": "1",
        "nodata": "0"
    }
    params = {
        "-ot": "Float32",
        "-of": "GTiff",
        "-co": "",
        "-zfield": "",
        "-z_increase": "",
        "-z_multiply": "",
        "-a_srs": "EPSG:32611",
        "-spat": "",
        "-clipsrc": "",
        "-clipsrcsql": "",
        "-clipsrclayer": "",
        "-clipsrcwhere": "",
        "-l": "",
        "-where": "",
        "-sql": "",
        "-txe": str(minx) + " " + str(maxx),
        "-tye": str(miny) + " " + str(maxy),
        # pixel is 0.5 meters
        "-outsize": str(deltx * 2) + " " + str(delty * 2),
        "-a": algo,
        "-q": "",
        "src_datasource": "",
        "dst_filename": ""
    }

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

    # 9 add result url to the database table
    # 10 log completed process (maybe should add the location of the result


    # 11 define the indexed array for the image

    # for clr in ["a"]:
    for clr in ["a", "b", "c", "d"]:
        d = gdal.Open(folder + id + "_" + clr + ".tiff")
        index, r_indexed, properties = gdalIO.raster_dataset_to_indexed_numpy(d, id, maxbands=1, bandLocation="byrow",
                                                                              nodata=0)
        print("saving indexed array to disk")
        fileIO.save_object(folder + id + "_indexedarray", (index, r_indexed, properties))
        d = None

    ######## rasterize rows with nearest neighbour

    data = {"layername": id + "_keep", "fullname": folder + id + "_keep.csv", "easting": "lon", "northing": "lat",
            "elevation": "row"}
    newxml = xml.format(**data)
    f = open(folder + id + "_keep_row.vrt", "w")
    f.write(newxml)
    f.close()

    algo = {
        "name": "nearest",
        "radius1": "0.4",
        "radius2": "0.4",
        "nodata": "0"
    }
    params = {
        "-ot": "Int16",
        "-of": "GTiff",
        "-co": "",
        "-zfield": "",
        "-z_increase": "",
        "-z_multiply": "",
        "-a_srs": "EPSG:32611",
        "-spat": "",
        "-clipsrc": "",
        "-clipsrcsql": "",
        "-clipsrclayer": "",
        "-clipsrcwhere": "",
        "-l": "",
        "-where": "",
        "-sql": "",
        "-txe": str(minx) + " " + str(maxx),
        "-tye": str(miny) + " " + str(maxy),
        "-outsize": str(deltx * 2) + " " + str(delty * 2),
        "-a": algo,
        "-q": "",
        "src_datasource": folder + id + "_keep_row.vrt",
        "dst_filename": folder + id + "_row.tiff"
    }

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
    row_indexed = gdalIO.apply_index_to_single_band(band, index)
    d = None

    # collect statistics for the comparisons
    stats = {}
    mean = []
    std = []

    # define new rasters increasing the data filtering and interpolating TODO: create configuration files for automation
    for step in [2, 3, 4]:

        # add dictionary for this step
        stats[step] = {}

        # filter data
        k, d = filter.filter_bystep(keep, step=step)

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

        # for clr in ["a"]:
        for clr in ["a", "b", "c", "d"]:
            data = {"layername": id + "_keep_step" + str(step),
                    "fullname": folder + id + "_keep_step" + str(step) + ".csv", "easting": "lon", "northing": "lat",
                    "elevation": clr}
            newxml = xml.format(**data)
            f = open(folder + id + "_keep" + clr + "_step" + str(step) + ".vrt", "w")
            f.write(newxml)
            f.close()

        # interpolate

        algo = {
            "name": "invdist",
            "radius1": str(0.5 * step),
            "radius2": str(0.5 * step),
            "angle": "0",
            "min_points": "1",
            "nodata": "0"
        }
        params = {
            "-ot": "Float32",
            "-of": "GTiff",
            "-co": "",
            "-zfield": "",
            "-z_increase": "",
            "-z_multiply": "",
            "-a_srs": "EPSG:32611",
            "-spat": "",
            "-clipsrc": "",
            "-clipsrcsql": "",
            "-clipsrclayer": "",
            "-clipsrcwhere": "",
            "-l": "",
            "-where": "",
            "-sql": "",
            "-txe": str(minx) + " " + str(maxx),
            "-tye": str(miny) + " " + str(maxy),
            "-outsize": str(deltx * 2) + " " + str(delty * 2),
            "-a": algo,
            "-q": "",
            "src_datasource": "",
            "dst_filename": ""
        }

        # for clr in ["a"]:
        for clr in ["a", "b", "c", "d"]:

            params["src_datasource"] = folder + id + "_keep" + clr + "_step" + str(step) + ".vrt"
            params["dst_filename"] = folder + id + "_" + clr + "_step" + str(step) + ".tiff"

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
            d = gdal.Open(folder + id + "_" + clr + "_step" + str(step) + ".tiff")
            band = d.GetRasterBand(1)
            # apply index
            new_r_indexed = gdalIO.apply_index_to_single_band(band, index)
            d = None

            # calculate average, std, total area (here a pixel is 0.25 m2,
            # totla area for pixel > 0, total area for pixels with 0 value


            stats[step][clr] = {}

            for i in np.unique(row_indexed):  # get the row numbers

                area = 0.25

                # add dictionary for this row
                stats[step][clr][i] = {}

                # get a mask for the current row
                mask = row_indexed == i
                # statistics for current row

                # r_indexed is 2d , while new_r_indexed and mask are 1d
                stats[step][clr][i]['truth'] = [r_indexed[0, :][mask].mean(), r_indexed[0, :][mask].std(),
                                                r_indexed[0, :][mask].shape[0] * area,
                                                np.count_nonzero(r_indexed[0, :][mask]) * area,
                                                r_indexed[0, :][mask][r_indexed[0, :][mask] == 0].shape[0] * area]
                stats[step][clr][i]['interpolated'] = [new_r_indexed[mask].mean(), new_r_indexed[mask].std(),
                                                       new_r_indexed[mask].shape[0] * area,
                                                       np.count_nonzero(new_r_indexed[mask]) * area,
                                                       new_r_indexed[mask][new_r_indexed[mask] == 0].shape[0] * area]
                # compare = r_indexed[0,:][mask] - new_r_indexed[mask]
                # stats[clr][step][i]['comparison'] = [compare.mean(),compare.std()]
                pass
                # calculate average and variance for this surface

                # mean.append(compare.mean())
                # std.append(compare.std())

    # save as a boxplot

    # print(mean)
    # print(std)
    return id, stats
    # {2: {"a": {1: {'truth': {}, 'interpolated': {}}}}}
# id, stats = interpolate_and_filter()
# folder = "/vagrant/code/pysdss/data/output/text/"
# fileIO.save_object( folder + id + "_statistics", stats)

##########################################################
##########################################################

#######################################################
##############berry workflow 1  # the following functions are for comparisons between dense data and filtered data
#### data is hardcoded because these are not production functions

def berrycolor_workflow_1(steps=[2,3,4], interp="invdist"):
    """
    similar to interpolate_and_filter but here as input i am using the original csv file and I am considering the
    number of berries as well
    
    steps: the filtering steps to use
    interp: the interpolation method to use on the filtered data   "invdist" or "average"
    
    
    :return:
    """

    if interp not in ["invdist","average"]:
        raise ValueError("Interpolation should be 'invdist' or 'average'")

    jsonpath = os.path.join(os.path.dirname(__file__), '../../../..', "pysdss/experiments/interpolation/")
    jsonpath = os.path.normpath(jsonpath)


    # 1 download filtered data with the chosen and create a dataframe

    folder = "/vagrant/code/pysdss/data/output/text/"
    experimentfolder = "/vagrant/code/pysdss/experiments/interpolation/"

    # create a folder to store the output
    id = utils.create_uniqueid()
    if not os.path.exists(folder+id):
        os.mkdir(folder+id)
    folder = folder+id + "/"


    # create a directory to store settings
    #if not os.path.exists(experimentfolder + str(id)):
    #   os.mkdir(experimentfolder + str(id))
    #dowload_data_test(id, folder)



    # dowload the data #TODO: create function to read from the database
    df = pd.read_csv("/vagrant/code/pysdss/data/input/2016_06_17.csv", usecols=["%Dataset_id"," Row"," Raw_fruit_count",
        " Visible_fruit_per_m"," Latitude"," Longitude","Harvestable_A","Harvestable_B","Harvestable_C","Harvestable_D"])
    df.columns = ["id","row","raw_fruit_count","visible_fruit_per_m", "lat", "lon","a","b","c","d"]

    #here should be necessary some mapping to get the column names


    # 2 filter data and save to disk . this is the "truth" so we just delete the zeros
    # 1/03/17  we don't filter
    #keep, discard = filter.filter_byvalue(df, 0, ">", colname="d")
    keep = df


    # 3 save filtered data to disk after reprojection
    # reproject latitude, longitude #WGS 84 / UTM zone 11N
    west,north = utils2.reproject_coordinates(keep, "epsg:32611", "lon", "lat", False)
    filter.set_field_value(keep, west, "lon")
    filter.set_field_value(keep, north, "lat")
    filter.set_field_value(keep, 0, "keepa")
    filter.set_field_value(keep, 0, "keepb")
    filter.set_field_value(keep, 0, "keepc")
    filter.set_field_value(keep, 0, "keepd")
    keep.to_csv(folder + id + "_keep.csv", index=False)


    '''west,north =utils.reproject_coordinates(discard, "epsg:32611", "lon", "lat", False)
    filter.set_field_value(discard, west,"lon")
    filter.set_field_value(discard, north,"lat")
    filter.set_field_value(discard, 0, "keepd")
    discard.to_csv(folder + id +"_discard.csv",index=False)'''


    # 4 create the virtualdataset header for the datafile

    with open ( jsonpath + "/vrtmodel.txt") as f:
        xml = f.read()

    # output the 4 virtual files for the 4 columns
    #for clr in ["a"]:
    for clr in ["a", "b", "c", "d"]:
        data = {"layername": id +"_keep", "fullname": folder + id +"_keep.csv", "easting":"lon", "northing":"lat", "elevation":clr}
        utils2.make_vrt(xml, data,folder + id+"_keep"+clr+".vrt" )

    # output the virtual files for raw fruit count and visible fruit count
    data = {"layername": id +"_keep", "fullname": folder + id +"_keep.csv", "easting":"lon", "northing":"lat", "elevation":"raw_fruit_count"}
    utils2.make_vrt(xml, data, folder + id+"_keep_rawfruit.vrt")

    data = {"layername": id +"_keep", "fullname": folder + id +"_keep.csv", "easting":"lon", "northing":"lat", "elevation":"visible_fruit_per_m"}
    utils2.make_vrt(xml, data, folder + id+"_keep_visiblefruit.vrt")


    # 5 define an interpolation grid (the user can choose an existing grid or draw on screen)
    #here i am doing manually for testing, i just take the bounding box

    minx = math.floor(keep['lon'].min())
    maxx = math.ceil(keep['lon'].max())
    miny = math.floor(keep['lat'].min())
    maxy = math.ceil( keep['lat'].max())
    delty = maxy - miny #height
    deltx = maxx - minx #width

    area_multiplier = 2 #2 to double the resulution and halve the pixel size to 0.5 meters

    # 6 log a running process to the database
    # 7 save grid metadata and interpolation metadata to the database historic tables
    # 8 execute gridding with gdal_array (set nodata value to zero or other value)


    path = jsonpath + "/average.json"
    params = utils.json_io(path, direction="in")

    params["-txe"]= str(minx) + " " + str(maxx)
    params["-tye"]= str(miny) + " " + str(maxy)
    # pixel is 0.5 meters
    params["-outsize"]= str(deltx * area_multiplier) + " " + str(delty * area_multiplier)

    #for clr in ["a"]:
    for clr in ["a", "b", "c", "d"]:

        params["src_datasource"] = folder + id+"_keep"+clr+".vrt"
        params["dst_filename"] = folder + id + "_" +clr + ".tiff"

        # build gdal_grid request
        text = grd.build_gdal_grid_string(params, outtext=False)
        print(text)
        text = ["gdal_grid"] + text
        print(text)

        #call gdal_grid
        print("Getting the 'truth' raster for color " + clr)
        out,err = utils.run_tool(text)
        print("output"+out)
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

    #for clr in ["a"]:
    for clr in ["a", "b", "c", "d"]:
        d = gdal.Open(folder + id+ "_" + clr + ".tiff")
        index, r_indexed, properties = gdalIO.raster_dataset_to_indexed_numpy(d, id, maxbands=1,bandLocation="byrow", nodata=-1)
        print("saving indexed array to disk")
        fileIO.save_object(folder + id+ "_indexedarray_"+clr, (index, r_indexed, properties))
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

    data = {"layername": id +"_keep", "fullname": folder + id +"_keep.csv", "easting":"lon", "northing":"lat", "elevation":"row"}
    utils2.make_vrt(xml, data, folder + id+"_keep_row.vrt")
    '''newxml = xml.format(**data)
    f = open(folder + id+"_keep_row.vrt", "w")
    f.write(newxml)
    f.close()'''

    path = jsonpath + "/nearest.json"
    params = utils.json_io(path, direction="in")

    params["-txe"]= str(minx) + " " + str(maxx)
    params["-tye"]= str(miny) + " " + str(maxy)
    params["-outsize"]= str(deltx * area_multiplier) + " " + str(delty * area_multiplier)
    params["src_datasource"]= folder + id + "_keep_row.vrt"
    params["dst_filename"]= folder + id + "_row.tiff"

    # build gdal_grid request
    text = grd.build_gdal_grid_string(params, outtext=False)
    print(text)
    text = ["gdal_grid"] + text
    print(text)

    #call gdal_grid
    print("Getting the row raster")
    out,err = utils.run_tool(text)
    print("output"+out)
    if err:
        print("error:" + err)
        raise Exception("gdal grid failed")

    # upload to numpy
    d = gdal.Open(folder + id+"_row.tiff")
    band = d.GetRasterBand(1)
    # apply index
    row_indexed = gdalIO.apply_index_to_single_band(band, index) # this is the last index from the fruit count, all the indexes are the same
    d = None

    #collect statistics for the comparisons
    stats={}
    #mean=[]
    #std=[]

    ####### get the indexed array for raw count
    index_raw, r_indexed_raw, properties_raw = fileIO.load_object(folder + id + "_indexedarray_rawfruit")
    ####### get the indexed array for visible count
    index_visible, r_indexed_visible, properties_visible = fileIO.load_object(
        folder + id + "_indexedarray_visiblefruit")

    # define new rasters increasing the data filtering and interpolating TODO: create configuration files for automation
    for step in steps:

        # add dictionary for this step
        stats[step] = {}

        # filter data by step
        k, d = filter.filter_bystep(keep, step=step)

        '''
        i already projected before
        west, north = utils.reproject_coordinates(k, "epsg:32611", "lon", "lat", False)
        filter.set_field_value(k, west, "lon")
        filter.set_field_value(k, north, "lat")
        filter.set_field_value(k, 0, "keepa")
        filter.set_field_value(k, 0, "keepb")
        filter.set_field_value(k, 0, "keepc")
        filter.set_field_value(k, 0, "keepd")'''
        k.to_csv(folder + id + "_keep_step"+str(step)+".csv", index=False)

        '''west, north = utils.reproject_coordinates(d, "epsg:32611", "lon", "lat", False)
        filter.set_field_value(d, west, "lon")
        filter.set_field_value(d, north, "lat")
        filter.set_field_value(d, 0, "keepa")
        filter.set_field_value(d, 0, "keepb")
        filter.set_field_value(d, 0, "keepc")
        filter.set_field_value(d, 0, "keepd")'''
        d.to_csv(folder + id + "_discard_step"+str(step)+".csv", index=False)

        #for clr in ["a"]:
        for clr in ["a", "b", "c", "d"]:

            data = {"layername": id + "_keep_step"+str(step), "fullname": folder + id + "_keep_step"+str(step)+".csv", "easting": "lon", "northing": "lat",
                    "elevation": clr}
            utils2.make_vrt(xml, data, folder + id + "_keep"+clr+"_step"+str(step)+".vrt")
            '''newxml = xml.format(**data)
            f = open(folder + id + "_keep"+clr+"_step"+str(step)+".vrt", "w")
            f.write(newxml)
            f.close()'''

        data = {"layername": id + "_keep_step" + str(step),
                "fullname": folder + id + "_keep_step" + str(step) + ".csv", "easting": "lon", "northing": "lat",
                "elevation": "raw_fruit_count"}
        utils2.make_vrt(xml, data, folder + id + "_keep_raw_step" + str(step) + ".vrt")
        '''newxml = xml.format(**data)
        f = open(folder + id + "_keep_raw_step" + str(step) + ".vrt", "w")
        f.write(newxml)
        f.close()'''

        data = {"layername": id + "_keep_step" + str(step),
                "fullname": folder + id + "_keep_step" + str(step) + ".csv", "easting": "lon", "northing": "lat",
                "elevation": "visible_fruit_per_m"}
        utils2.make_vrt(xml, data, folder + id + "_keep_visible_step" + str(step) + ".vrt")
        '''newxml = xml.format(**data)
        f = open(folder + id + "_keep_visible_step" + str(step) + ".vrt", "w")
        f.write(newxml)
        f.close()'''

        # interpolate
        if interp== "invdist":
            path = jsonpath + "/invdist.json"
        if interp== "average":
            path = jsonpath + "/average.json"

        params = utils.json_io(path, direction="in")

        params["-txe"]= str(minx) + " " + str(maxx)
        params["-tye"]= str(miny) + " " + str(maxy)
        params["-outsize"]= str(deltx * area_multiplier) + " " + str(delty * area_multiplier)
        params["-a"]["radius1"] = str(0.5 * step)
        params["-a"]["radius2"] = str(0.5 * step)


        for clr in ["raw", "visible"]:

            params["src_datasource"] = folder + id + "_keep_" + clr + "_step" + str(step) + ".vrt"
            params["dst_filename"] = folder + id + "_" + clr + "_step" + str(step) + ".tiff"

            print(params)
            # build gdal_grid request
            text = grd.build_gdal_grid_string(params, outtext=False)
            print(text)
            text = ["gdal_grid"] + text
            print(text)

            # call gdal_grid
            print("Getting filtered raster by step " + str(step) + "for color " + clr)
            out, err = utils.run_tool(text)
            print("output" + out)
            if err:
                print("error:" + err)
                raise Exception("gdal grid failed")


        # upload to numpy
        d = gdal.Open(folder + id + "_raw_step" + str(step) + ".tiff")
        band = d.GetRasterBand(1)
        # apply index
        new_r_indexed_raw = gdalIO.apply_index_to_single_band(band, index)# this is the last index from the fruit count
        d = None

        d = gdal.Open(folder + id + "_visible_step" + str(step) + ".tiff")
        band = d.GetRasterBand(1)
        # apply index
        new_r_indexed_visible = gdalIO.apply_index_to_single_band(band, index)# this is the last index from the fruit count
        d = None

        #for clr in ["a"]:
        for clr in ["a", "b", "c", "d"]:

            params["src_datasource"] = folder + id + "_keep" + clr + "_step" + str(step)+".vrt"
            params["dst_filename"] = folder + id + "_"+clr + "_step"+str(step)+".tiff"

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
            d = gdal.Open(folder + id + "_"+clr + "_step"+str(step)+".tiff")
            band = d.GetRasterBand(1)
            # apply index
            new_r_indexed = gdalIO.apply_index_to_single_band(band, index)# this is the last index from the fruit count
            d = None

            # calculate average, std, total area (here a pixel is 0.25 m2,
            # totla area for pixel > 0, total area for pixels with 0 value

            ####### get the indexed array for this colour grade

            #d = gdal.Open(folder + id + "_" + clr + ".tiff")
            #band = d.GetRasterBand(1)
            #r_indexed = gdalIO.apply_index_to_single_band(band, index) # this is the last index from the fruit count
            #d = None
            index, r_indexed, properties= fileIO.load_object(folder + id + "_indexedarray_" + clr)


            stats[step][clr] = {}

            for i in np.unique(row_indexed): #get the row numbers

                area = math.pow(1/area_multiplier,2)

                #add dictionary for this row
                stats[step][clr][i] = {}

                #get a mask for the current row
                mask = row_indexed == i
                #statistics for current row

                #average, std, totalarea,total nonzero area, total zeroarea, total raw fruit count,
                # average raw fruit count, std raw fruit count ,total visible fruitxm, average visible fruitXm, std visible fruitXm

                #r_indexed is 2d , while new_r_indexed and mask are 1d
                stats[step][clr][i]['truth'] = [r_indexed[0,:][mask].mean(), r_indexed[0,:][mask].std(),
                                                r_indexed[0,:][mask].shape[0]*area, np.count_nonzero(r_indexed[0,:][mask])*area,
                                                r_indexed[0,:][mask][r_indexed[0,:][mask] == 0].shape[0]*area,
                                                r_indexed_raw[0,:][mask].sum(),
                                                r_indexed_raw[0,:][mask].mean(),
                                                r_indexed_raw[0,:][mask].std(),
                                                r_indexed_visible[0,:][mask].sum(),
                                                r_indexed_visible[0,:][mask].mean(),
                                                r_indexed_visible[0,:][mask].std()]

                '''
                stats[step][clr][i]['truth'] = [r_indexed[mask].mean(), r_indexed[mask].std(),
                                                r_indexed[mask].shape[0]*area, np.count_nonzero(r_indexed[mask])*area,
                                                r_indexed[mask][r_indexed[mask] == 0].shape[0]*area,
                                                r_indexed_raw[0,:][mask].sum(), r_indexed_raw[0,:][mask].mean(),r_indexed_raw[0,:][mask].std(),
                                                r_indexed_visible[0,:][mask].sum(),r_indexed_visible[0,:][mask].mean()], r_indexed_visible[0,:][mask].std()]'''

                stats[step][clr][i]['interpolated'] = [new_r_indexed[mask].mean(), new_r_indexed[mask].std(),
                                                       new_r_indexed[mask].shape[0]*area, np.count_nonzero(new_r_indexed[mask])*area,
                                                       new_r_indexed[mask][new_r_indexed[mask] == 0].shape[0]*area,
                                                       new_r_indexed_raw[mask].sum(),
                                                       new_r_indexed_raw[mask].mean(),
                                                       new_r_indexed_raw[mask].std(),
                                                       new_r_indexed_visible[mask].sum(),
                                                       new_r_indexed_visible[mask].mean(),
                                                       new_r_indexed_visible[mask].std()]

                #compare = r_indexed[0,:][mask] - new_r_indexed[mask]
                #stats[clr][step][i]['comparison'] = [compare.mean(),compare.std()]
                #pass
        # calculate average and variance for this surface

        #mean.append(compare.mean())
        #std.append(compare.std())

    #save as a boxplot

    #print(mean)
    #print(std)
    return id,stats
    #{2: {"a": {1: {'truth': {}, 'interpolated': {}}}}}

def test_interpret_result(stats, id, folder):
    '''
    Interpret the dictionary with the results and save a table

    table columns are:
    row|avg|std|area|nozero_area|zero_area|raw_count|raw_avg|raw_std|visible_count|visible_avg|visible_std
    step2avg|step2std|step2area|step2nozero_area|step2zero_area|step2raw_count|step2raw_avg|step2raw_std|step2visible_count|step2visible_avg|step2visible_std
    |step3.....

    :param stats:
        dictionary in the form
        {step:{"colorgrade": { row : {'truth':{},'interpolated':{} }} } }
    :param rsme: True to calculate rootmeansquare error
    :return: None
    '''

    # get the number of steps and their value
    nsteps = len(stats)
    print(nsteps)
    b = [i for i in stats]
    b.sort()
    print(b)

    # define the column names and convert to string
    colmodel = ["avg", "std", "area","nozero_area", "zero_area", "raw_count","raw_avg","raw_std","visible_count","visible_avg","visible_std"]
    NCOL = len(colmodel)
    cols =  ["row"] + colmodel
    for s in b:
        cols += ["step"+str(s)+"_"+i for i in colmodel]
    print(cols)
    print(len(cols))


    # calculate the number of columns for the result
    ncolumns = NCOL * (nsteps + 1) + 1
    print(ncolumns)

    # get the name of the colorgrades
    c = [i for i in stats[b[0]]]
    c.sort()
    print(c)

    #get the number of rows
    nrows = len(stats[b[0]][c[0]])
    print(nrows)

    #step=1
    for j in c: #for each colorgrade we want to save in a separate file

        # create a numpy array to store result
        output = np.zeros((nrows, ncolumns), dtype=np.float32)

        # first step, color grade j; output both the truth and the first step
        f = stats[b[0]][ j ]

        r = 0 if 0 in f else 1 #if the column starts at 1 we need to know (the array index start at zero)
        for row in f:  # iterate rows, each one has truth and interpolated

            nrow = row-r #correct the index for the rows

            output[nrow, 0] = row

            truth = f[row]["truth"]
            for i in range(NCOL):
                output[nrow, i+1] = truth[i]

            interp = f[row]["interpolated"]
            for i in range(NCOL):
                output[nrow, i + 1 + NCOL] = interp[i]

        # now we can output the remaining steps
        if nsteps >1:
            for n in range(1, nsteps):
                f = stats[b[n]][j]
                for row in f:
                    nrow = row - r #correct the index for the rows
                    interp = f[row]["interpolated"]
                    for i in range(NCOL):
                        output[nrow, i + 1 + (NCOL*2) + NCOL*(n-1)] = interp[i] # 1row + 5truth + 5firststep + other steps

        outcols = ','.join(cols)
        #save array to disk as a csv for this color grade
        np.savetxt(folder + id +"_"+j+"_output.csv",output , fmt='%.3f', delimiter=',', newline='\n', header=outcols, footer='', comments='')


def calculate_rmse_row(folder,id, nsteps=[2,3,4,5,6,7,8,9,10], colorgrades = ["a","b","c","d"]):
    """
    rmse based on comparison with the rasterized points
    """
    #open a random csv as pandas
    path = folder + id + "_" + random.choice(colorgrades) + "_output.csv"
    file = pd.read_csv(path)

    outtext = ""

    # calculate rmse for visible count (this is the same for all colour grade files)
    for e,c in enumerate(nsteps):
        outtext += "step"+str(c)+"_visible_avg\t" + str(utils.rmse( file["step"+str(c)+"_visible_avg"].values, file["visible_avg"].values)) + "\n"

    for e,c in enumerate(colorgrades):
        #for each step
        path = folder + id + "_" + c + "_output.csv"
        file = pd.read_csv(path)
        for j,x in enumerate(nsteps):

            outtext += "step" + str(x) + "_avg\t"+ str(utils.rmse(file["step" + str(x) + "_avg"].values, file["avg"].values)) + "\n"

    with open(folder + id + "_rmse_row.csv","w") as f:
        f.write(outtext)


def calculate_rmse(folder,id, nsteps=[2,3,4,5,6,7,8,9,10], colorgrades = ["a","b","c","d"]):
    """
    rmse based on comparison with the original points (this is preferred to calculate_rmse_row)
    :param folder: 
    :param id: 
    :param nsteps: 
    :param colorgrades: 
    :return: 
    """

    rowname = "row"
    stepcount = len(nsteps)
    colorcount = len(colorgrades)

    path = folder + id + "_ERRORSTATISTICS.csv"
    df = pd.read_csv(path)

    #get name unique rows and initialize an empty array for the output
    rows = df[rowname].unique()
    arr = np.zeros((len(rows), 2 + stepcount+ colorcount + stepcount*colorcount))

    #iterate the row
    for i,row in enumerate(rows):
        # select one vineyard row
        fdf = df[df[rowname] == row]

        #set value for row and average visible_fruit_per_m for the current row
        arr[i,0] = row
        arr[i, 1] = fdf['visible_fruit_per_m'].mean()

        # calculate rmse for visible count
        for e, c in enumerate(nsteps):
            arr[i,2 + e] = utils.rmse(fdf["step" + str(c) + "visible"].values, fdf["visible_fruit_per_m"].values)

        #set average value for colorgrades for the current row
        for e,c in enumerate(colorgrades):
            idxstart = 2 + stepcount
            arr[i, idxstart + e] = fdf[c].mean()

        # calculate root mean square error for colorgrades
        for e,c in enumerate(colorgrades):
            idxstart = 2 + stepcount + colorcount + (e * stepcount)
            for j, x in enumerate(nsteps):
                arr[i, idxstart + j] = utils.rmse(fdf["step" + str(x) + c].values, fdf[c].values)

    #set xolumn names and save as a csv file
    cols = ['row', 'visible_avg']
    cols += ["visible_step"+str(s)+"_rmse" for s in nsteps]
    cols +=  [i + "_avg" for i in colorgrades]
    cols += ["step" + str(i) + "_" + c + "_rmse" for c in colorgrades for i in nsteps]
    outcols = ','.join(cols)
    np.savetxt(folder + id + "_rsme_row.csv",arr , fmt='%.3f', delimiter=',', newline='\n', header=outcols, footer='', comments='')


def test_compare_raster_totruth(point, id, folder,epsg=32611, step=[2, 3,4,5,6,7,8,9,10], removeduplicates=True, clean=None):

        # 1 open  csv

        #id = "a16b2a828b9174d678e76be46619eb329"
        #folder = "/vagrant/code/pysdss/data/output/text/" + id + "/"
        #point = folder + id + "_keep.csv"

        if removeduplicates:
            # remove duplicates and save to newfile, set name for point file
            df = utils.remove_duplicates(point, ["lat", "lon"], operation="mean")
            point = folder + id + "_keep_nodup.csv"
            df.to_csv(point)
        else:
            df = pd.read_csv(point, usecols=["id", "row", "a", "b", "c", "d", "raw_fruit_count", "visible_fruit_per_m"])

        # 2 extract
        #step = [2, 3, 4,10]
        colorgrade = ["a", "b", "c", "d"]
        counts = ["raw", "visible"]

        # define a list of files
        rasters = [folder + id + "_" + y + "_step" + str(x) + ".tiff" for x in step for y in colorgrade]
        rasters2 = [folder + id + "_" + y + "_step" + str(x) + ".tiff" for x in step for y in counts]

        spatialRef = osr.SpatialReference()
        spatialRef.ImportFromEPSG(epsg)
        srs = spatialRef.ExportToProj4()
        vrtdict = {"layername": os.path.basename(point).split(".")[0], "fullname": point, "easting": "lon",
                   "northing": "lat", "srs": srs}

        c = []
        for r in rasters:
            stat = utils2.compare_csvpoint_cell(point, r, vrtdict, overwrite=False)
            c.append(stat)

        arr = np.transpose(np.asarray(c))
        newdf = pd.DataFrame(arr, columns=["step" + str(x) + y for x in step for y in colorgrade])

        c = []
        for r in rasters2:
            stat = utils2.compare_csvpoint_cell(point, r, vrtdict, overwrite=False)
            c.append(stat)

        arr = np.transpose(np.asarray(c))
        newdf2 = pd.DataFrame(arr, columns=["step" + str(x) + y for x in step for y in counts])

        outdf = pd.concat([df, newdf, newdf2], axis=1)

        if clean: # clean the dataframe, useful to hide nodata values in the result  e.g convert -1 with 0
            utils.clean_dataframe(outdf, clean[0], clean[1]) # TODO check the use of  DataFrame.replace

        outdf.to_csv(folder + id + "_ERRORSTATISTICS.csv")


def estimate_berries_byrow(id, folder, step=[2, 3, 4, 5, 6, 7, 8, 9, 10]):

    """ Output comparison between original points and interploated points for average, std, estimated number per row
    :param id: 
    :param folder: 
    :param step: 
    :return: 
    """

    left_fields = ["row","length","visible_m_avg_original","visible_m_std_original","estimated_berries_original"]
    right_fields = ["visible_m_avg", "delta_avg", "visible_std", "delta_std", "estimated_berries", "delta_estimates"]


    # calculate row length
    lengths = utils.row_length(folder + id + "_keep.csv", "lon", "lat", "row", conversion_factor=1)
    # initialize output array
    out = np.zeros((lengths.shape[0], 5 + (len(steps)+1)*6))
    out[:,0:2] = lengths

    # open  _ERRORSTATISTICS and calculate average and std per row for the visible count
    path = folder + id + "_ERRORSTATISTICS.csv"
    df = pd.read_csv(path, usecols=['row', 'visible_fruit_per_m'])

    # open one of the files with the statistics for the rasterized data
    path = folder + id + "_d_output.csv"
    df2 = pd.read_csv(path)

    #gey statistics for original points and rasterized points
    rows = lengths[:,0]
    for i,row in enumerate(rows):
        # select one vineyard row
        fdf = df[df['row'] == row]

        out[i, 2] = fdf['visible_fruit_per_m'].mean()
        out[i, 3] = fdf['visible_fruit_per_m'].std()
        out[i, 4] = out[i, 1] * out[i, 2]

        out[i, 5] = df2["visible_avg"].values[i]
        out[i, 6] = np.absolute(out[i, 5] - out[i, 2])
        out[i, 7] = df2["visible_std"].values[i]
        out[i, 8] = np.absolute(out[i, 7] - out[i, 3])
        out[i, 9] = out[i, 1] * out[i, 5]
        out[i, 10] = np.absolute(out[i, 9] -  out[i, 4])

    # get statistics for interpolated filtered points
    idxstart = 11
    for s in step:
        for i, row in enumerate(rows):
            out[i, idxstart] = df2[ "step"+str(s)+"_visible_avg"].values[i]
            out[i, idxstart+1] = np.absolute(out[i, idxstart] - out[i, 2])
            out[i, idxstart+2] = df2["step"+str(s)+"_visible_std"].values[i]
            out[i, idxstart+3] = np.absolute(out[i, idxstart+2] - out[i, 3])
            out[i, idxstart+4] = out[i, 1] * out[i, idxstart]
            out[i, idxstart+5] = np.absolute(out[i, idxstart+4] - out[i, 4])
        idxstart += len(right_fields)

    # save array to disk
    cols = left_fields + right_fields
    cols += ["step"+str(x)+"_"+ i  for x in step for i in right_fields ]
    outcols = ','.join(cols)
    np.savetxt(folder + id + "_count_stats_byrow.csv", out, fmt='%.3f', delimiter=',', newline='\n', header=outcols, footer='',
               comments='')


if __name__ == "__main__":
    pass

    steps=[2,3,4,5,6,7,8,9,10]
    #id, stats = berrycolor_workflow_1(steps,"average")
    #id, stats = berrycolor_workflow_1(steps,"average")
    #folder = "/vagrant/code/pysdss/data/output/text/"+id+"/"
    #fileIO.save_object( folder + id + "_statistics", stats)
    #id = "a8aa8edad1c9d457b826f3e156edf0fae"
    #folder = "/vagrant/code/pysdss/data/output/text/" +id + "/"
    #stats = fileIO.load_object(folder + id + "_statistics")
    #test_interpret_result(stats,id,folder)
    #id = "a5d75134d88a843f388af925670d89772"
    #folder = "/vagrant/code/pysdss/data/output/text/" + id + "/"
    #point = folder + id + "_keep.csv"
    #test_compare_raster_totruth(point,"a0857fd6fbb0e4f6fbacb26b1a984779a", folder,step=steps )
    #test_compare_raster_totruth(point, id, folder,step=steps, clean=[[-1],[0]])


    #id = "a9dd672655abb4188b6d8ac2cc830b3e4"
    #folder = "/vagrant/code/pysdss/data/output/text/workflow1/inversedistancetopower/190517/"+ id + "/"
    #point = folder + id + "_keep.csv"
    #test_compare_raster_totruth(point, id, folder, step=steps, clean=[[-1], [0]])
    #calculate_rmse(folder, id)

    id = "a5d75134d88a843f388af925670d89772"
    folder = "/vagrant/code/pysdss/data/output/text/workflow1/movingaverage/"+ id + "/"
    estimate_berries_byrow(id, folder)
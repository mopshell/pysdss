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
import shutil
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

def berrycolor_workflow_1_original(steps=[2,3,4], interp="invdist"):
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


def berrycolor_workflow_1(steps=[2, 3, 4], interp="invdist", random_filter=False, first_last=False, force_interpolation_by_row=False, rowdirection="x"):
    """
    similar to interpolate_and_filter but here as input i am using the original csv file and I am considering the
    number of berries as well

    steps: the filtering steps to use
    interp: the interpolation method to use on the filtered data   "invdist" or "average", kriging will be added later
    random filter:   true to take random points (e,g step2 takes 50% random points for each row, false use ordered filter
    (e.g. step2 takes 1 point every 2 points)
    first_last: true to always have the first and last point of a vineyard row
    force_interpolation : true if i want to be sure the points belong to the current row only
    :return:
    """

    if interp not in ["invdist", "average"]:
        raise ValueError("Interpolation should be 'invdist' or 'average'")

    jsonpath = os.path.join(os.path.dirname(__file__), '../../../..', "pysdss/experiments/interpolation/")
    jsonpath = os.path.normpath(jsonpath)

    # 1 download filtered data with the chosen and create a dataframe

    folder = "/vagrant/code/pysdss/data/output/text/"
    experimentfolder = "/vagrant/code/pysdss/experiments/interpolation/"

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
    west, north = utils2.reproject_coordinates(keep, "epsg:32611", "lon", "lat", False)
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

    # collect statistics for the comparisons
    stats = {}
    # mean=[]
    # std=[]

    ####### get the indexed array for raw count
    index_raw, r_indexed_raw, properties_raw = fileIO.load_object(folder + id + "_indexedarray_rawfruit")
    ####### get the indexed array for visible count
    index_visible, r_indexed_visible, properties_visible = fileIO.load_object(
        folder + id + "_indexedarray_visiblefruit")

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

        # define new rasters increasing the data filtering and interpolating TODO: create configuration files for automation
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

            # for clr in ["a"]:
            for clr in ["a", "b", "c", "d"]:
                data = {"layername": id + "_keep_step" + str(step),
                        "fullname": folder + id + "_keep_step" + str(step) + ".csv", "easting": "lon", "northing": "lat",
                        "elevation": clr}
                utils2.make_vrt(xml, data, folder + id + "_keep" + clr + "_step" + str(step) + ".vrt")
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
            if interp == "invdist":
                path = jsonpath + "/invdist.json"
            if interp == "average":
                path = jsonpath + "/average.json"

            params = utils.json_io(path, direction="in")

            params["-txe"] = str(minx) + " " + str(maxx)
            params["-tye"] = str(miny) + " " + str(maxy)
            params["-outsize"] = str(deltx * area_multiplier) + " " + str(delty * area_multiplier)
            if random_filter: #if we have random points we set we set the radius to the max distance between points plus a buffer of 0.5
                params["-a"]["radius1"] = str(max_point_distance + 0.5)
                params["-a"]["radius2"] = str(max_point_distance + 0.5)
            else:
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
            new_r_indexed_raw = gdalIO.apply_index_to_single_band(band,
                                                                  index)  # this is the last index from the fruit count
            d = None

            d = gdal.Open(folder + id + "_visible_step" + str(step) + ".tiff")
            band = d.GetRasterBand(1)
            # apply index
            new_r_indexed_visible = gdalIO.apply_index_to_single_band(band,
                                                                      index)  # this is the last index from the fruit count
            d = None

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
                                                    r_indexed_raw[0, :][mask].sum(),
                                                    r_indexed_raw[0, :][mask].mean(),
                                                    r_indexed_raw[0, :][mask].std(),
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
                                                           new_r_indexed_raw[mask].sum(),
                                                           new_r_indexed_raw[mask].mean(),
                                                           new_r_indexed_raw[mask].std(),
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



def test_interpret_result(stats, id, folder, colmodel=None):
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
    if not colmodel:
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
            row = int(row)
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


def calculate_rmse_row_0(folder,id, nsteps=[2,3,4,5,6,7,8,9,10], colorgrades = ["a","b","c","d"]):
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


def calculate_rmse_row(folder,id, nsteps=[2,3,4,5,6,7,8,9,10], colorgrades = ["a","b","c","d"]):
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


def calculate_rmse(folder,id,nsteps=[2,3,4,5,6,7,8,9,10]):
    """
    Calculate the root mean square error for the visible per meter average and the estimated berries . The differences
    are calculated per number of vineyard rows and the number of items are the rows
    :param folder: 
    :param id: 
    :param nsteps: 
    :return: 
    """

    #create pandas dataframe
    path = folder + id + "_count_stats_byrow.csv"
    df = pd.read_csv(path)

    # number of vineyard rows
    nrows = df.shape[0]

    # store the output field named
    outcols_0 = []
    outcols_1 = []

    #cretae empty numpy array
    arr = np.zeros((1, len(steps)*2))

    #iterate the steps and fill in the array
    for i,s in enumerate(nsteps):

        #rmse for visible_avg, add to the output array, append the column name
        x =  math.sqrt((np.sum(np.power(df["step"+str(s)+"_delta_avg"].values,2))/nrows))
        arr[0, i] = x
        outcols_0.append("step"+str(s)+"_delta_avg_rmse")
        #rmse for estimated berries, add to the output array, append the column name
        y = math.sqrt((np.sum(np.power(df["step"+str(s)+"_delta_estimates"].values, 2)) / nrows))
        arr[0, i+len(steps)] = y
        outcols_1.append("step"+str(s)+"_delta_estimates_rmse")

    outcols_0 += outcols_1
    outcols = ','.join(outcols_0)
    #export array to disk
    np.savetxt(folder + id + "_rsme.csv",arr , fmt='%.3f', delimiter=',', newline='\n', header=outcols, footer='', comments='')

def test_compare_raster_totruth(point, id, folder,epsg=32611, step=[2, 3,4,5,6,7,8,9,10], removeduplicates=True, clean=None,
                                colorgrade=None, counts=None):
        """
        comparing original point values with the underlying pixel values and save comparison to disk
        use this when the workflow was using force_interpolation_by_row=False
        :param point: this is the id +"_keep.csv" file
        :param id: operation id
        :param folder: the folder with point
        :param epsg: epsg code for spatial reference
        :param step: the steps to consider
        :param removeduplicates: true to get average value for duplicate points
        :param clean: chage first value with the second [[-1],[0]] , useful to clean nodata values
        :param tempfolder: name of the folder storing the rasters by row
        :return:
        """

        #step = [2, 3, 4,10]

        if not colorgrade:
            colorgrade = ["a", "b", "c", "d"]
        if not counts:
            counts = ["raw", "visible"]

        # 1 open  csv

        #id = "a16b2a828b9174d678e76be46619eb329"
        #folder = "/vagrant/code/pysdss/data/output/text/" + id + "/"
        #point = folder + id + "_keep.csv"


        if removeduplicates:#TODO check that remove dulicates will not alter the row number
            # remove duplicates and save to newfile, set name for point file
            df = utils.remove_duplicates(point, ["lat", "lon"], operation="mean")
            point = folder + id + "_keep_nodup.csv"
            df.to_csv(point)

        df = pd.read_csv(point, usecols=["id", "row", "a", "b", "c", "d", "raw_fruit_count", "visible_fruit_per_m"])

        # 2 extract


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


def test_compare_raster_totruth_byrow(point, id, folder, epsg=32611, step=[2, 3, 4, 5, 6, 7, 8, 9, 10], removeduplicates=True,
                                clean=None, tempfolder="/temprasters/", colorgrade=None, counts=None):

    """
    comparing original point values with the underlying pixel values and save comparison to disk
    use this when the workflow was using force_interpolation_by_row=True

    :param point: this is the id +"_keep.csv" file
    :param id: operation id
    :param folder: the folder with point
    :param epsg: epsg code for spatial reference
    :param step: the steps to consider
    :param removeduplicates: true to get average value for duplicate points
    :param clean: chage first value with the second [[-1],[0]] , useful to clean nodata values
    :param tempfolder: name of the folder storing the rasters by row
    :return:
    """


    # step = [2, 3, 4,10]
    if not colorgrade:
        colorgrade = ["a", "b", "c", "d"]
    if not counts:
        counts = ["raw", "visible"]

    # 1 open  csv
    # id = "a16b2a828b9174d678e76be46619eb329"
    # folder = "/vagrant/code/pysdss/data/output/text/" + id + "/"
    # point = folder + id + "_keep.csv"

    if removeduplicates:  # TODO check that remove dulicates will not alter the row number
        # remove duplicates and save to newfile, set name for point file
        df = utils.remove_duplicates(point, ["lat", "lon"], operation="mean")
        point = folder + id + "_keep_nodup.csv"
        df.to_csv(point, index=False)

    df = pd.read_csv(point, usecols=["id", "row", "a", "b", "c", "d", "raw_fruit_count", "visible_fruit_per_m","lat","lon"])
    # find unique row numbers
    rows = sorted(df["row"].unique())
    #filter by row and save to tempfolder
    for row in rows:
        row = int(row)
        rowdf = df[df["row"] == row]
        # save csv file to disk
        rowdf.to_csv(folder + "/temprasters/" + id + "_keep_row" + str(row) + ".csv", index=False)
    df.drop(["lat", "lon"] , axis=1, inplace=True) #we dont want in the final result

    # 2 extract
    # define a dictionary of files
    rasters = {}
    rasters2 = {}
    for row in rows:
        row = int(row)
        #this will be in order a,b,c,d   and raw,visible
        rasters[row] = [folder + tempfolder + id + "_" + y + "_step" + str(x) + "_row" + str(row) + ".tiff" for x in step for y in colorgrade]
        rasters2[row] = [folder + tempfolder + id + "_" + y + "_step" + str(x) + "_row" + str(row) + ".tiff" for x in step for y in counts]

    #define a gdal spatial reference object
    spatialRef = osr.SpatialReference()
    spatialRef.ImportFromEPSG(epsg)
    srs = spatialRef.ExportToProj4()

    c = []
    for row in rows:
        row = int(row)
        orig_points = folder + "/temprasters/" + id + "_keep_row" + str(row) + ".csv"
        vrtdict = {"layername": os.path.basename(orig_points).split(".")[0], "fullname": orig_points, "easting": "lon",
                   "northing": "lat", "srs": srs}
        raster_row = []
        for r in rasters[row]:
            stat = utils2.compare_csvpoint_cell(orig_points, r, vrtdict, overwrite=True)
            raster_row.append(stat) #this will be a linear collecion of values step2a, step2b,....
        c.append(np.transpose(np.asarray(raster_row)))

    # we stack vertically all the rows and convert to dataframe
    arr_colour = np.vstack(c)
    newdf = pd.DataFrame(arr_colour, columns=["step" + str(x) + y for x in step for y in colorgrade])

    c = []
    for row in rows:
        row = int(row)
        orig_points = folder + "/temprasters/" + id + "_keep_row" + str(row) + ".csv"
        vrtdict = {"layername": os.path.basename(orig_points).split(".")[0], "fullname": orig_points, "easting": "lon",
                   "northing": "lat", "srs": srs}
        raster_row = []
        for r in rasters2[row]:
            #overwrite false because we already have the shapefiles
            stat = utils2.compare_csvpoint_cell(orig_points, r, vrtdict, overwrite=False)
            raster_row.append(stat)
        c.append(np.transpose(np.asarray(raster_row)))

    arr_count = np.vstack(c)
    newdf2 = pd.DataFrame(arr_count, columns=["step" + str(x) + y for x in step for y in counts])

    #stack the 3 dataframe horizontally
    outdf = pd.concat([df, newdf, newdf2], axis=1)

    if clean:  # clean the dataframe, useful to hide nodata values in the result  e.g convert -1 with 0
        utils.clean_dataframe(outdf, clean[0], clean[1])  # TODO check the use of  DataFrame.replace

    outdf.to_csv(folder + id + "_ERRORSTATISTICS.csv", index=False)



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


def chart_rmse(folder,id):
    """Chart the rmse
    :param folder: 
    :param id: 
    :return: 
    """

    from bokeh.plotting import figure
    from bokeh.io import output_file,  save

    #open pandas dataframe
    path = folder + id + "_rsme.csv"
    df = pd.read_csv(path)

    ncols = df.shape[1]
    #the categorical labels for x axis (step2,step3,step4,....)
    x_labels = [i.split("_")[0] for i in df.columns][:int(len(df.columns) / 2)]

    #make new chart directory if it does not exist
    if not os.path.exists(folder +"/charts"):
        os.mkdir(folder + "/charts")

    #chart ["stepx_delta_avg_rmse"]
    output_file(folder + "/charts/delta_avg_rmse.html",title="delta_avg_rmse")
    f = figure(x_range=x_labels)
    f.line(x_labels , df.values[0,0:int(ncols/2)], line_width=2)
    save(f)

    #chart ("stepx_delta_estimates_rmse")
    output_file(folder + "/charts/delta_estimates_rmse.html",title="delta_estimates_rmse")
    f = figure(x_range=x_labels)
    f.line(x_labels , df.values[0,int(ncols/2):], line_width=2)
    save(f)

def chart_estimated_berries(id, folder, steps=[2]):

    """
    Chart the estimated berries
    :param id:
    :param folder:
    :param steps:
    :return:
    """
    from bokeh.plotting import figure
    from bokeh.io import output_file,  save

    path = folder + id + "_count_stats_byrow.csv"

    cols = ['row', 'estimated_berries'] + ["step"+str(s)+"_estimated_berries" for s in steps]

    df = pd.read_csv(path, usecols=cols)

    #make new chart directory if it does not exist
    if not os.path.exists(folder +"/charts"):
        os.mkdir(folder + "/charts")

    #make new chart directory if it does not exist
    if not os.path.exists(folder +"/charts/estimatedberries"):
        os.mkdir(folder + "/charts/estimatedberries")

    for s in steps:

        output_file(folder +"/charts/estimatedberries/estimated_step"+str(s)+".html")
        f = figure(title='Estimeted berries: original VS step'+str(s))
        f.line(df["row"], df["estimated_berries"], line_color="green", line_width=1, legend="Original")
        f.line(df["row"], df["step"+str(s)+"_estimated_berries"], line_color="red", line_width=1, legend="Step"+str(s))
        f.legend.location = "top_left"
        save(f)






if __name__ == "__main__":
    pass

    #steps=[2,3,4,5,6,7,8,9,10]


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
    #calculate_rmse_row(folder, id)

    #id = "a5d75134d88a843f388af925670d89772"
    #folder = "/vagrant/code/pysdss/data/output/text/workflow1/movingaverage/"+ id + "/"
    #estimate_berries_byrow(id, folder)


    #id = "a9dd672655abb4188b6d8ac2cc830b3e4"
    #folder = "/vagrant/code/pysdss/data/output/text/workflow1/inversedistancetopower/190517/" + id + "/"
    #estimate_berries_byrow(id, folder)
    #calculate_rmse(folder, id)

    #id = "a5d75134d88a843f388af925670d89772"
    #folder = "/vagrant/code/pysdss/data/output/text/workflow1/movingaverage/" + id + "/"
    #estimate_berries_byrow(id, folder)
    #calculate_rmse(folder, id)

    ##output plots
    #id = "a5d75134d88a843f388af925670d89772"
    #folder = "/vagrant/code/pysdss/data/output/text/workflow1/movingaverage/" + id + "/"
    #chart_rmse(folder,id)



    '''
    #############################workflow1 updated
    steps=[2,3,4,5,6,7,8,9,10,20,30,50,100]


    # original worflow with ordered filter and overlapping interpolation
    id, stats = berrycolor_workflow_1(steps,"average")
    folder = "/vagrant/code/pysdss/data/output/text/"+id+"/"
    fileIO.save_object( folder + id + "_statistics", stats)
    test_interpret_result(stats, id, folder)
    point = folder + id + "_keep.csv"
    test_compare_raster_totruth(point, id, folder, step=steps, clean=[[-1], [0]])
    calculate_rmse_row(folder, id, nsteps=steps)
    estimate_berries_byrow(id, folder,step=steps)
    calculate_rmse(folder,id,nsteps=steps)
    chart_rmse(folder, id)

    id, stats = berrycolor_workflow_1(steps)
    folder = "/vagrant/code/pysdss/data/output/text/"+id+"/"
    fileIO.save_object( folder + id + "_statistics", stats)
    test_interpret_result(stats, id, folder)
    point = folder + id + "_keep.csv"
    test_compare_raster_totruth(point, id, folder, step=steps, clean=[[-1], [0]])
    calculate_rmse_row(folder, id, nsteps=steps)
    estimate_berries_byrow(id, folder, step=steps)
    calculate_rmse(folder, id, nsteps=steps)
    chart_rmse(folder, id)
    #######################################

    # worflow with random filter (with first_last row point) and overlapping interpolation
    id, stats = berrycolor_workflow_1(steps, "average",random_filter=True, first_last=True)
    folder = "/vagrant/code/pysdss/data/output/text/" + id + "/"
    fileIO.save_object(folder + id + "_statistics", stats)
    test_interpret_result(stats, id, folder)
    point = folder + id + "_keep.csv"
    test_compare_raster_totruth(point, id, folder, step=steps, clean=[[-1], [0]])
    calculate_rmse_row(folder, id, nsteps=steps)
    estimate_berries_byrow(id, folder, step=steps)
    calculate_rmse(folder, id, nsteps=steps)
    chart_rmse(folder, id)

    # worflow with random filter (with first_last row point) and overlapping interpolation
    id, stats = berrycolor_workflow_1(steps, "invdist", random_filter=True, first_last=True)
    folder = "/vagrant/code/pysdss/data/output/text/" + id + "/"
    fileIO.save_object(folder + id + "_statistics", stats)
    test_interpret_result(stats, id, folder)
    point = folder + id + "_keep.csv"
    test_compare_raster_totruth(point, id, folder, step=steps, clean=[[-1], [0]])
    calculate_rmse_row(folder, id, nsteps=steps)
    estimate_berries_byrow(id, folder, step=steps)
    calculate_rmse(folder, id, nsteps=steps)
    chart_rmse(folder, id)
    #######################################


    # workflow with ordered filter  and force interpolation by row
    id, stats = berrycolor_workflow_1(steps, "average",force_interpolation_by_row=True)
    folder = "/vagrant/code/pysdss/data/output/text/" + id + "/"
    fileIO.save_object(folder + id + "_statistics", stats)
    stats = fileIO.load_object(folder + id + "_statistics")
    test_interpret_result(stats,id,folder)
    point = folder + id + "_keep.csv"
    test_compare_raster_totruth_byrow(point, id, folder,step=steps, clean=[[-1],[0]])
    calculate_rmse_row(folder, id, nsteps=steps)
    estimate_berries_byrow(id, folder, step=steps)
    calculate_rmse(folder, id, nsteps=steps)
    chart_rmse(folder, id)

    # workflow with ordered filter  and force interpolation by row
    id, stats = berrycolor_workflow_1(steps, "invdist",force_interpolation_by_row=True)
    folder = "/vagrant/code/pysdss/data/output/text/" + id + "/"
    fileIO.save_object(folder + id + "_statistics", stats)
    stats = fileIO.load_object(folder + id + "_statistics")
    test_interpret_result(stats,id,folder)
    point = folder + id + "_keep.csv"
    test_compare_raster_totruth_byrow(point, id, folder,step=steps, clean=[[-1],[0]])
    calculate_rmse_row(folder, id, nsteps=steps)
    estimate_berries_byrow(id, folder, step=steps)
    calculate_rmse(folder, id, nsteps=steps)
    chart_rmse(folder, id)

    ################################

    # workflow with random filter (with first_last row point)  and force interpolation by row
    id, stats = berrycolor_workflow_1(steps, "average",force_interpolation_by_row=True,random_filter=True, first_last=True)
    folder = "/vagrant/code/pysdss/data/output/text/" + id + "/"
    fileIO.save_object(folder + id + "_statistics", stats)
    stats = fileIO.load_object(folder + id + "_statistics")
    test_interpret_result(stats,id,folder)
    point = folder + id + "_keep.csv"
    test_compare_raster_totruth_byrow(point, id, folder,step=steps, clean=[[-1],[0]])
    calculate_rmse_row(folder, id, nsteps=steps)
    estimate_berries_byrow(id, folder, step=steps)
    calculate_rmse(folder, id, nsteps=steps)
    chart_rmse(folder, id)

    # workflow with random filter (with first_last row point)  and force interpolation by row
    id, stats = berrycolor_workflow_1(steps, "invdist",force_interpolation_by_row=True,random_filter=True, first_last=True)
    folder = "/vagrant/code/pysdss/data/output/text/" + id + "/"
    fileIO.save_object(folder + id + "_statistics", stats)
    stats = fileIO.load_object(folder + id + "_statistics")
    test_interpret_result(stats,id,folder)
    point = folder + id + "_keep.csv"
    test_compare_raster_totruth_byrow(point, id, folder,step=steps, clean=[[-1],[0]])
    calculate_rmse_row(folder, id, nsteps=steps)
    estimate_berries_byrow(id, folder, step=steps)
    calculate_rmse(folder, id, nsteps=steps)
    chart_rmse(folder, id)
    '''

    steps = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]
    id= "a40b9fc7181184bf0b9835f830a677513"
    folder = "/vagrant/code/pysdss/data/output/text/analysis0806_steps2to100/" + id + "/"
    chart_estimated_berries(id, folder, steps=steps)




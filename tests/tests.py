# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        tests.py
# Purpose:     code for testing
#
# Author:      claudio piccinini
#
# Created:     08/02/2017
#-------------------------------------------------------------------------------

import pandas as pd
import math
import json
import shutil
import warnings

import os
import os.path
from osgeo import gdal
from osgeo import gdalconst as gdct
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array as gdar

import numpy as np
from scipy.interpolate import Rbf
from scipy.misc import imsave

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


from rasterstats import point_query

from pysdss.utility import fileIO
from pysdss.utility import gdalRasterIO as gdalIO

from pysdss import utils
from pysdss.database import database as dt
from pysdss.filtering import filter
from pysdss.gridding import gridding as grd
from pysdss.utility import utils as utils2

connstring = {"dbname": "grapes", "user": "claudio", "password": "claudio", "port": "5432", "host": "127.0.0.1"}
datasetid = 26
boundid = 1


if __name__ == "__main__":

    ##################################################################################
    #######################  pysdss.database tests  ##################################
    ##################################################################################

    ########## download generic 2 joined data table
    def dowload_data_test(id, path):
        """
        dowload data for a single color grade
        :param id:
        :param path:
        :return:
        """
        leftclm = ["row", "id_sdata", "lat", "lon"]
        rightclmn = ["d", "keepd::int"]

        outfile = path + id + ".csv"
        # complete dataset
        dt.download_data("id_sdata", "data.SensoDatum", "id_sensor", leftclm, "id_sdata", "data.colorgrade", rightclmn,
                         datasetid, outfile,orderclms=["row", "id_sdata"], conndict=connstring)
        name = utils.create_uniqueid()
        outfile = "/vagrant/code/pysdss/data/output/" + name + ".csv"
        # clipped dataset, result ordered
        dt.download_data("id_sdata", "data.SensoDatum", "id_sensor", leftclm, "id_sdata", "data.colorgrade", rightclmn,
                         datasetid, outfile, leftboundgeom="geom", boundtable="data.boundary", righttboundgeom="geom",
                         boundid="id_bound",
                         boundvalue=1, orderclms=["row", "id_sdata"], conndict=connstring)



    #dowload_data_test(utils.create_uniqueid(), "/vagrant/code/pysdss/data/output/text/" )


    ########## download generic 2 joined data table
    def dowload_all_data_test(id, path):
        """
        download data for all 4 grades
        :param id:
        :param path:
        :return:
        """
        leftclm = ["row", "id_sdata", "lat", "lon"]
        rightclmn = ["a","b","c","d","keepa::int","keepb::int","keepc::int","keepd::int"]

        outfile = path + id + ".csv"
        # complete dataset
        dt.download_data("id_sdata", "data.SensoDatum", "id_sensor", leftclm, "id_sdata", "data.colorgrade", rightclmn,
                         datasetid, outfile,orderclms=["row", "id_sdata"], conndict=connstring)
        name = utils.create_uniqueid()
        outfile = "/vagrant/code/pysdss/data/output/" + name + ".csv"
        # clipped dataset, result ordered
        dt.download_data("id_sdata", "data.SensoDatum", "id_sensor", leftclm, "id_sdata", "data.colorgrade", rightclmn,
                         datasetid, outfile, leftboundgeom="geom", boundtable="data.boundary", righttboundgeom="geom",
                         boundid="id_bound",
                         boundvalue=1, orderclms=["row", "id_sdata"], conndict=connstring)



    ##############################################################################
    ######################  pysdss.filtering tests  ##############################
    ##############################################################################
    '''

    #open file
    n= "aee6cf556fd2a4835a0e5d7f4806c6665"
    folder = "/vagrant/code/pysdss/data/output/text/"
    df = pd.read_csv( folder + n+".csv")
    #strip and lower column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lower()
    print(df.columns)


    #filter data
    #for each row
    #filter by passing  a step
    #filter by percentage
    #filter by threshold value

    # filter by step
    #keep, discard = filter.filter_bystep(df,step=20,rowname="row" )
    # filter by value
    #keep, discard = filter.filter_byvalue(df, 0,">", colname="d",rowname="row")
    #filter by range, this show how to concatenate discard dataframe over multiple oprations
    '''

    '''keep, discard= filter.filter_byvalue(df, 0.8,"<=",colname="d")
    if not(keep.empty): keep,discard2 =filter.filter_byvalue(keep, 0.3,">=", colname="d")
    discard = pd.concat([discard,discard2])
    discard = discard.sort_values(["row","id_sdata"])'''

    '''
    #test wrong operator
    #keep, discard = filter.filter_byvalue(df, 0.8, "!", colname="d")
    #test filter by std, do not consider zeros
    keep, discard = filter.filter_byvalue(df, 0, ">", colname="d")
    if not (keep.empty): keep, discard2, stats = filter.filter_bystd(keep,colname="d", rowname="row", backstats=True)
    print(stats)
    discard = pd.concat([discard,discard2])
    discard = discard.sort_values(["row","id_sdata"])

    #save filtered data to disk after reprojection
    # reproject latitude, longitude #WGS 84 / UTM zone 11N
    west,north =utils.reproject_coordinates(keep, "epsg:32611", "lon", "lat", False)
    filter.set_field_value(keep, west,"lon")
    filter.set_field_value(keep, north,"lat")
    filter.set_field_value(keep, 0, "keepd")
    keep.to_csv(folder + n+"_keep.csv",index=False)

    west,north =utils.reproject_coordinates(discard, "epsg:32611", "lon", "lat", False)
    filter.set_field_value(discard, west,"lon")
    filter.set_field_value(discard, north,"lat")
    filter.set_field_value(discard, 0, "keepd")
    discard.to_csv(folder + n+"_discard.csv",index=False)

    #clust.kmeans_cluster(newdf, west,north)
    #print(df.shape)
    #newdf=filter_byvalue(df, 0.8,"<=","row", "harvestable_d")
    #if not(newdf.empty): newdf2=filter_rows_byvalue(newdf, 0.3,">=", "row", "harvestable_c")
    #if not(newdf2.empty):
    #    west, north = reproject_column(newdf2, "32611", "longitude", "latitude", False)
    #    clust.kmeans_cluster(newdf2, west,north)
    #else: print("no values!")'''

    #print(df.shape)
    #newdf=filter_bystd(df, nstd=1, colname= "harvestable_d", userows=True)
    #print(newdf.shape)
    #west, north = change_coordinates(newdf, "32611", "longitude", "latitude", False)
    #clust.kmeans_cluster(newdf, west,north,4)

    #newdf = pd.DataFrame(np.c_[west,north, df['longitude'].values, df['latitude'].values,kmeans.labels_], columns=["x", "y", "long", "lat","label"])


    #histogram use the pandas charts???
    #n, bins, patches = plt.hist(df["harvestable_a"].values, 50, normed=0, facecolor='green', alpha=0.75)
    #plt.show()



    ###############################



    def test_rasterize_points(folder, id):

        df = pd.read_csv(folder + id + "_keep.csv")

        minx = math.floor(df['lon'].min())
        maxx = math.ceil(df['lon'].max())
        miny = math.floor(df['lat'].min())
        maxy = math.ceil(df['lat'].max())
        delty = maxy - miny
        deltx = maxx - minx


        xml = '''<OGRVRTDataSource>
                    <OGRVRTLayer name="{layername}">
                        <SrcDataSource>{fullname}</SrcDataSource>
                        <GeometryType>wkbPoint</GeometryType>
                        <GeometryField encoding="PointFromColumns" x="{easting}" y="{northing}" />
                    </OGRVRTLayer>
        </OGRVRTDataSource>
        '''

        # output the 4 virtual files for the 4 columns
        # for clr in ["a"]:

        data = {"layername": id + "_keep", "fullname": folder + id + "_keep.csv", "easting": "lon", "northing": "lat"}
        utils2.make_vrt(xml, data, folder + id + "_keep_rasterized_truth.vrt")
        '''newxml = xml.format(**data)
        f = open(folder + id + "_keep_rasterized_truth.vrt", "w")
        f.write(newxml)
        f.close()'''


        params = {
            "-b": "",
            "-i": "",
            "-at": "",
            "-burn":"1",
            "-a": "",
            "-3d": "",
            "-add": "",
            "-l": "",
            "-where": "",
            "-sql": "",
            "-dialect": "",
            "-of": "",
            "-a_srs":"EPSG:32611",
            "-co": "",
            "-a_nodata": "0",
            "-init": "",
            "-te": str(minx) + " " + str(miny) + " " + str(maxx) + " " + str(maxy) ,
            "-tr": "",
            "-tap": "",
            "-ts": str(deltx * 2) + " " + str(delty * 2),
            "-ot": "Byte",
            "-q": "",
            "src_datasource": "",
            "dst_filename": ""
        }

        params["src_datasource"] = folder + id + "_keep_rasterized_truth.vrt"
        params["dst_filename"] = folder + id + "_keep_rasterized_truth.tiff"

        # build gdal_grid request
        text = grd.build_gdal_rasterize_string(params, outtext=False)
        print(text)
        text = ["gdal_rasterize"] + text
        print(text)

        # call gdal_rasterize
        print("Getting the 'truth' raster for full set")
        out, err = utils.run_tool(text)
        print("output" + out)
        if err:
            print("error:" + err)
            raise Exception("gdal rasterize failed")


        for j in ["2","3","4"]:
            data = {"layername": id + "_keep_step"+j, "fullname": folder + id + "_keep_step"+j+".csv", "easting": "lon", "northing": "lat"}
            utils2.make_vrt(xml, data, folder + id + "_keep_rasterized_step"+j+".vrt")
            '''newxml = xml.format(**data)
            f = open(folder + id + "_keep_rasterized_step"+j+".vrt", "w")
            f.write(newxml)
            f.close()'''

        for j in ["2", "3", "4"]:

            params["src_datasource"] = folder + id + "_keep_rasterized_step"+j+".vrt"
            params["dst_filename"] = folder + id + "_keep_rasterized_step"+j+".tiff"

            # build gdal_grid request
            text = grd.build_gdal_rasterize_string(params, outtext=False)
            print(text)
            text = ["gdal_rasterize"] + text
            print(text)

            # call gdal_rasterize
            print("Getting the 'truth' raster " + j)
            out, err = utils.run_tool(text)
            print("output" + out)
            if err:
                print("error:" + err)
                raise Exception("gdal rasterize failed")


    id = "a4cc04d563b944d769a1513842356ed36"
    folder = "/vagrant/code/pysdss/data/output/text/"
    #rasterize_points(folder,id)

    def test_convert_to_binary():
        id = "a4cc04d563b944d769a1513842356ed36"
        folder = "/vagrant/code/pysdss/data/output/text/"


        text = ["gdal_translate","-ot","Byte",folder + id + "_row.tiff",folder + id + "_row_binary.tiff"]
        print(text)

        out, err = utils.run_tool(text)
        print("output" + out)
        if err:
            print("error:" + err)
            raise Exception("gdal grid failed")

    #convert_to_binary()

    def test_simplefilter():
        """
        :return:

        {'a': {'full': {1:[average, std,min,max], 2:.....}, 'step2': : {1:[average, std,min,max], 2:.....},...},
         'b': {.....}, 'c': {.....}, 'd': {} }


        """

        #download data
        folder = "/vagrant/code/pysdss/data/output/text/"
        experimentfolder = "/vagrant/code/pysdss/experiments/interpolation/"

        id = utils.create_uniqueid()

        # create a directory to store settings
        #if not os.path.exists(experimentfolder + str(id)):
        #   os.mkdir(experimentfolder + str(id))


        dowload_all_data_test(id, folder)

        df = pd.read_csv(folder + id + ".csv")
        # strip and lower column names
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.lower()

        #calculate statistics by row

        result = {}

        for n in ["a", "b","c","d"]:

            stats = filter.calculate_stat_byrow(df, rowname="row", colname=n)
            a = {"full":{}}
            for i in stats: a['full'].update(i)

            # for increasing steps

            for i in [2,3,4]:
                # calculate statistics by row
                keep, delete = filter.filter_bystep(df, step=i)

                stats = filter.calculate_stat_byrow(keep, rowname="row", colname=n)

                b = {"step"+str(i): {}}
                for j in stats: b["step"+str(i)].update(j)

                a.update(b)

            result[n] = a

        return result

    #print(simplefilter())


    #step = 4 # here could be an iteration to increase the step

    #keep, discard = filter.filter_bystep(df,step=step)
    #discard = pd.concat([discard,discard2])
    #discard = discard.sort_values(["row","id_sdata"])

    # interpolate the filtered data
    # use the index to extract the pixels
    # compare the pixel values to create some sort of difference index
    # save the diffrence image to disk



############old workflow 2###############

    def test_interpret_result2(stats,folder):
        '''
        Interpret the dictionary with the results from berrycolor_workflow_2 and save a table

        table columns are:
        row|avg|std|area|nozero_area|zero_area|raw_count|raw_avg|raw_std|visible_count|visible_avg|visible_std

        :param stats:
            dictionary in the form

            {"colorgrade": {row: [0.031, 0.029, 93.0, 83.75, 9.25, 118339.1, 318.11, 29.281, 213405.12, 573.66968, 61.072674],

        :return: None
        '''

        #get the number of colors (this will be the number of output files)
        ncolors = len(stats) #4
        colors = [i for i in stats] #a,b,c,d

        #get the number of rows
        nrows = len(stats[colors[0]])

        #check if rows start with 0 or 1, this will be needed to fill in the array
        rowfix = 1
        if 0 in stats[colors[0]]: rowfix=0

        #get the number of columns
        # define the column names and convert to string
        colmodel = ["avg", "std", "area","nozero_area", "zero_area", "raw_count","raw_avg","raw_std","visible_count","visible_avg","visible_std"]
        cols =  ["row"] + colmodel
        NCOL = len(cols)
        cols=','.join(cols)

        # create a numpy array to store result
        output = np.zeros((nrows, NCOL), dtype=np.float32)

        # iterate by color
        for color in colors:
            for row in stats[color]:
                output[row-rowfix,:] = [row]+ stats[color][row] #select the array row to fill in based on the row number
            #save array to disk as a csv for this color grade
            np.savetxt(folder +"/_"+color+"_output.csv",output , fmt='%.3f', delimiter=',', newline='\n', header=cols, footer='', comments='')
            output.fill(0)


    def berrycolor_workflow_2_old(file, folder="/vagrant/code/pysdss/data/output/text/", gdal=True):
        """
        testing interpolation and statistics along the line (this was the old workflow_2 for sparse data, when there
        was also the workflow_3, now there is only a workflow 2) (see colordata/colordata.py)

        use gdal False for scipy radial basis
        :return:
        """

        ############################ 1 download filtered data with the chosen and create a dataframe
        # create a folder to store the output
        id = utils.create_uniqueid()
        if not os.path.exists(folder+id):
            os.mkdir(folder+id)
        folder = folder+id + "/"

        #set the path to folder with settings
        jsonpath = os.path.join(os.path.dirname(__file__), '..', "pysdss/experiments/interpolation/")
        jsonpath = os.path.normpath(jsonpath)

        #1 convert point to polyline

        utils2.csv_to_polyline_shapefile(file, ycol="lat", xcol="lon", linecol="row", epsg=32611 , outpath=folder + "rows.shp")

        #############################  2 buffer the polyline

        utils2.buffer(folder + "rows.shp", folder + "rows_buffer.shp", 0.25)

        ##############################  3rasterize poliline

        #need a grid
        df = pd.read_csv(file)
        minx = math.floor(df['lon'].min())
        maxx = math.ceil(df['lon'].max())
        miny = math.floor(df['lat'].min())
        maxy = math.ceil( df['lat'].max())
        delty = maxy - miny #height
        deltx = maxx - minx #width

        area_multiplier = 2 #2 to double the resulution and halve the pixel size to 0.5 meters

        path = jsonpath + "/rasterize.json"
        params = utils.json_io(path, direction="in")

        params["-a"]= "id_row"
        params["-te"]= str(minx) + " " + str(miny) + " " + str(maxx) + " " + str(maxy)
        params["-ts"]= str(deltx * 2) + " " + str(delty * 2) #pixel 0,5 meters
        params["-ot"]= "Int16"

        params["src_datasource"] = folder + "rows_buffer.shp"
        params["dst_filename"] = folder + "rows_buffer.tiff"

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

        #################################   4get buffered poliline index

        d = gdal.Open( folder + "rows_buffer.tiff")
        row_index, row_indexed, row_properties = gdalIO.raster_dataset_to_indexed_numpy(d, id, maxbands=1, bandLocation="byrow",
                                                                              nodata=-1)

        print("saving indexed array to disk")
        fileIO.save_object( folder +  "rows_buffer_index", (row_index, row_indexed, row_properties))
        d = None

        ###################################   5interpolate points, use the index to extract statistics along the line

        with open ( jsonpath + "/vrtmodel.txt") as f:
            xml = f.read()

        # output the 4 virtual files for the 4 columns
        for clr in ["a", "b", "c", "d"]:
            data = {"layername": os.path.basename(file).split(".")[0], "fullname": file, "easting":"lon", "northing":"lat", "elevation":clr}
            utils2.make_vrt(xml, data,folder + "_"+clr+".vrt" )
        # output the 2 virtual files for or raw fruit count and visible fruit count
        data = {"layername": os.path.basename(file).split(".")[0], "fullname": file, "easting":"lon", "northing":"lat", "elevation":"raw_fruit_count"}
        utils2.make_vrt(xml, data, folder + "_rawfruit.vrt")
        data = {"layername": os.path.basename(file).split(".")[0], "fullname": file, "easting":"lon", "northing":"lat", "elevation":"visible_fruit_per_m"}
        utils2.make_vrt(xml, data, folder + "_visiblefruit.vrt")


        # interpolate

        if gdal:
            path = jsonpath + "/invdist.json"
            params = utils.json_io(path, direction="in")
            params["-txe"] = str(minx) + " " + str(maxx)
            params["-tye"] = str(miny) + " " + str(maxy)
            params["-outsize"] = str(deltx * area_multiplier) + " " + str(delty * area_multiplier)
            params["-a"]["radius1"] = "10"
            params["-a"]["radius2"] = "10"
            params["-a"]["smoothing"] = "20"
            params["-a"]["power"] = "0"

        else: #scipy
            #set up the interpolation grid
            tix = np.linspace(minx, maxx, deltx*2)
            tiy = np.linspace(miny, maxy, delty*2)
            XI, YI = np.meshgrid(tix, tiy)

        if gdal:
            for clr in ["raw", "visible"]:

                params["src_datasource"] = folder + "_"+ clr +"fruit.vrt"
                params["dst_filename"] = folder + "_"+ clr + "fruit.tiff"

                #print(params)
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
        else:
            for clr in ["raw_fruit_count","visible_fruit_per_m"]:
                rbf = Rbf(df['lon'].values, df['lat'].values, df[clr].values)
                ZI = rbf(XI, YI)
                print()


        # upload to numpy
        d = gdal.Open(folder + "_rawfruit.tiff")
        band = d.GetRasterBand(1)
        # apply index
        new_r_indexed_raw = gdalIO.apply_index_to_single_band(band,row_index)
        d = None

        d = gdal.Open(folder + "_visiblefruit.tiff")
        band = d.GetRasterBand(1)
        # apply index
        new_r_indexed_visible = gdalIO.apply_index_to_single_band(band,row_index)
        d = None

        stats = {}


        for clr in ["a", "b", "c", "d"]:
            params["src_datasource"] = folder + "_"+clr+".vrt"
            params["dst_filename"] =folder +  "_" + clr + ".tiff"

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
            d = gdal.Open(folder + "_" + clr + ".tiff")
            band = d.GetRasterBand(1)
            # apply index
            new_r_indexed = gdalIO.apply_index_to_single_band(band, row_index)  # this is the index from the rows
            d = None

            stats[clr] = {}

            for i in np.unique(row_indexed):  # get the row numbers

                area = math.pow(1 / area_multiplier, 2)

                # get a mask for the current row
                mask = row_indexed == i
                # statistics for current row

                # average, std, totalarea,total nonzero area, total zeroarea, total raw fruit count,
                # average raw fruit count, std raw fruit count ,total visible fruitxm, average visible fruitXm, std visible fruitXm

                # r_indexed is 2d , while new_r_indexed and mask are 1d

                stats[clr][i] = [ new_r_indexed[mask[0,:]].mean(), new_r_indexed[mask[0,:]].std(),
                                                       new_r_indexed[mask[0,:]].shape[0] * area,
                                                       np.count_nonzero(new_r_indexed[mask[0,:]]) * area,
                                                       new_r_indexed[mask[0,:]][new_r_indexed[mask[0,:]] == 0].shape[0] * area,
                                                       new_r_indexed_raw[mask[0,:]].sum(),
                                                       new_r_indexed_raw[mask[0,:]].mean(),
                                                       new_r_indexed_raw[mask[0,:]].std(),
                                                       new_r_indexed_visible[mask[0,:]].sum(),
                                                       new_r_indexed_visible[mask[0,:]].mean(),
                                                       new_r_indexed_visible[mask[0,:]].std() ]

        return id,stats



    #folder = "/vagrant/code/pysdss/data/output/text/"
    #file = folder + "/aa6b4aece6bcb494f82f4c3d56ab47dd8/aa6b4aece6bcb494f82f4c3d56ab47dd8_keep_step4.csv"
    #id,stats = berrycolor_workflow_1(file,folder)
    #test_interpret_result(stats, id, folder)
    #print(stats)
    #utils.json_io(folder+str(id)+"/_stats.json",stats )
    #fileIO.save_object( folder+str(id)+"/_stats", stats)
    #TODO function to convert to csv
    #stats = fileIO.load_object( folder+str(id)+"/_stats")
    #test_interpret_result2(stats, folder+id)


    def test_radial():

        folder = "/vagrant/code/pysdss/data/output/text/"
        file = folder + "/workflow1/aa6b4aece6bcb494f82f4c3d56ab47dd8_keep_step4.csv"

        #need a grid
        df0 = pd.read_csv(file)

        df = utils.remove_duplicates(df0, ['lat','lon'])

        minx = math.floor(df['lon'].min())
        maxx = math.ceil(df['lon'].max())
        miny = math.floor(df['lat'].min())
        maxy = math.ceil( df['lat'].max())
        delty = maxy - miny #height
        deltx = maxx - minx #width

        tix = np.linspace(minx, maxx, deltx/2,dtype=np.float32) #pixel 1m, with pixel 0.5 would fail
        tiy = np.linspace(miny, maxy, delty/2,dtype=np.float32)
        XI, YI = np.meshgrid(tix, tiy)

        #for clr in ["raw_fruit_count", "visible_fruit_per_m"]:
        for clr in ["raw_fruit_count"]:

            x = df['lon'].values.astype(np.float32)
            y = df['lat'].values.astype(np.float32)
            z = df[clr].values.astype(np.float32)

            rbf = Rbf(x,y,z, smooth=1)
            ZI = rbf(XI, YI)
            print("here")

            #plt.imshow(ZI)
            ##plt.subplot(1, 1, 1)
            ##plt.pcolor(XI, YI, ZI, cmap=cm.jet)
            ##plt.scatter(df['lon'].values, df['lat'].values, 100, df[clr].values, cmap=cm.jet)
            ##plt.title('RBF interpolation - multiquadrics')
            ##plt.colorbar()
            ##plt.savefig(folder+'/rbf2d.png')

            imsave(folder+'/rbf2d.png', ZI)

    #test_radial()


##########oldworkflow2################


###########testing reclassify a raster with a treshold


# todo make function to download the data

# todo can we reproject on the database before downloading?


# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        gridding_tests.py
# Purpose:     code for interpolation testing
#
# Author:      claudio piccinini
#
# Created:     24/02/2017
#-------------------------------------------------------------------------------
import os
from osgeo import gdal
from osgeo import gdalconst as gdct
from pysdss.utility import fileIO
from pysdss.utility import gdalRasterIO as gdalIO




if __name__ == "__main__":

    folder = "/vagrant/code/pysdss/data/output/text/"
    os.chdir(folder)

    #################### test for raster properties
    def describeRaster(rasterpath):
        d = None
        try:
            d = gdal.Open(rasterpath)
            return gdalIO.describe_raster_dataset(d, os.path.basename(rasterpath))

        except Exception as e:
            print(e)
        finally:
            if d is not None:
                d = None  # give memory control back to C++ code


    #name, width, height, number of bands, coordinate system string, geotransform)
    #print(describeRaster("acffabe6629804827a13295bc5023245e.tiff"))

    ####################### test for 1 band
    def importSingleBand(rasterpath):
        d = None
        band = None
        try:
            d = gdal.Open(rasterpath)
            band = d.GetRasterBand(1)
            print ("exporting to indexed numpy array")
            a = gdalIO.single_band_to_indexed_numpy(band,nodata=0)
            print ("saving object to disk")
            fileIO.save_object( os.getcwd()+ "/singlebandtest", a)
            print ("loading object from disk")
            c=fileIO.load_object(os.getcwd()+ "/singlebandtest")
            return a,c

        except Exception as e:
            print (e)
        finally:
            if d is not None:
                d = None
            if band is not None:
                band = None

    #print(importSingleBand("acffabe6629804827a13295bc5023245e.tiff"))

    # test for multiband
    def importRaster(rasterpath):
        d = None
        try:
            print(os.getcwd())
            d = gdal.Open(rasterpath)
            b = gdalIO.raster_dataset_to_indexed_numpy(d, os.path.basename(rasterpath), maxbands = 10, bandLocation="bycolumn", nodata= 0)
            print("saving object to disk")
            fileIO.save_object(os.getcwd()+ "/multibandtest", b)
            print("loading object from disk")
            c = fileIO.load_object(os.getcwd()+ "/multibandtest")
            return b, c

        except Exception as e:
            print(e)
        finally:
            if d is not None:
                d = None


    #b,c = importRaster("acffabe6629804827a13295bc5023245e.tiff")

    # test for exporting indexed array
    def exportRaster(obj):
        try:
            print("loading object from disk")
            c = fileIO.load_object(obj)
            dataset = gdalIO.indexed_numpy_to_raster_dataset(c, "exported.tif", outfolder=os.getcwd(), datatype=gdct.GDT_Float32, nodata=0, returnlist=True, frmt="GTiff")
            return dataset

        except Exception as e:
            print(e)


    exportRaster("multibandtest")
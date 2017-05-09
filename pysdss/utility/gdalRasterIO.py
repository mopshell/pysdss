# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:     gdalRasterIO
# Purpose:  - describe a raster dataset
#           - import a sinle raster band to an indexed 2D numpy array
#           - import a multiband raster to an indexed 2D numpy array
#           - import a list of rasters to an indexed 2D numpy array (only the first bands)
#           - export a numpy array to a raster
#           - create a a list where each element as a 1dimensional array for each raster band
#
# Author:   claudio piccinini,
#
#
# Created:     06/03/2015
# -------------------------------------------------------------------------------

# import sys
# set module path for pydoc documentation
# sys.path.append(r"")

import numpy as np
from osgeo import gdal
from osgeo import gdal_array as gdar
from osgeo import gdalconst as gdct


gdal.UseExceptions() # allow gdal exceptions


def describe_raster_dataset(rasterdataset, name):
    """Get the properties of a gdal raster dataset
    :param rasterdataset:
    :param name: a gdal raster dataset, the dataset name
    :return: return the properties of a raster dataset ( name, width, height, number of bands, coordinate system string , geotransform)
    """

    width = rasterdataset.RasterXSize
    height = rasterdataset.RasterYSize
    nbands = rasterdataset.RasterCount
    csystem = rasterdataset.GetProjection()
    geotransform = rasterdataset.GetGeoTransform()
    name = name

    return name, width, height, nbands, csystem, geotransform


def single_band_to_indexed_numpy(band, nodata=-3.40282e+038, return_index=True):
    """ Convert a dataset band into an indexed array

    :param band: a gdal band
    :param nodata: pixel no data value
    :param return_index: a bool to return the array index
    :return: (index, r_indexed)  or  r_indexed , depending on return_index
    """

    try:
        # convert a gdal band to a numpyarray
        px = gdar.BandReadAsArray(band, 0, 0, band.XSize, band.YSize)
        index = np.nonzero(px > nodata)  # define the index
        r_indexed = px[index]  # apply the index to the band data
        if return_index:
            return index, r_indexed
        else:
            return r_indexed
    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e



def apply_index_to_single_band(band, index, return_index=False):
    """ Apply an existing index to a band

    :param band: a gdal band
    :param nodata: a numpy index
    :param return_index: a bool to return the array index
    :return: (index, r_indexed)  or  r_indexed , depending on return_index
    """

    try:
        # convert a gdal band to a numpyarray
        px = gdar.BandReadAsArray(band, 0, 0, band.XSize, band.YSize)

        r_indexed = px[index]  # apply the index to the band data
        if return_index:
            return index, r_indexed
        else:
            return r_indexed
    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e


def raster_dataset_to_indexed_numpy(dataset, name, maxbands=100, bandLocation="bycolumn", nodata=-3.40282e+038):

    """Convert a multiband raster dataset into an indexed array

       important: we supposed the index is the same for all the bands, and all the bands have same number
       of rows columns, the nodata value have the same location

    :param dataset: gdal raster dataset
    :param name: the raster name
    :param maxbands: the number of bands to import
    :param bandLocation: how to store bands (possible values: "byrow" , "bycolumn")
    :param nodata: pixel no data value
    :return: the index, the 2d indexed array , dictionary with raster properties
    """

    if maxbands < 1 : raise Exception("number of bands should be >=1!!!")
    if not (bandLocation == "byrow" or bandLocation == "bycolumn"):
        raise Exception("you need the specify the correct \"bandLocation\" argument!!!")
    try:
        # describe the band and set a dictionary with the raster properties
        name, width, height, nbands, csystem, geotransform = describe_raster_dataset(dataset, name)
        properties={"name": name, "columns": width,"rows":height, "nbands": nbands,"csystem": csystem, "geotransform": geotransform, "bands orientation": bandLocation, "bands": {}}
        properties["bands"] = {"columns": [], "rows": [], "blocksize": [], "nodatavalue": []}

        # set the number of bands to process
        nbands = properties["nbands"]
        print("This raster has "+ str(nbands) + " bands")
        if maxbands <= nbands:
            nbands = maxbands

        # initialize a list of indexed arrays
        r_indexed = list()  # TODO improve memory usage

        # iterate the bands and get an index for the first band and the indexed arrays
        # we get the index for the first band only (we supposed this is the same for all the bands, 
        # and all bands have same number of rows columns)
        print("We are processing " + str(nbands) + " bands")
        return_index = True
        for idx in range(1, nbands + 1): #gdal bands starts at 1
            band = dataset.GetRasterBand(idx)
            if band is None:
                continue
            print("processing band "+str(idx))
            bandarray = single_band_to_indexed_numpy(band, nodata, return_index)
            # we get the index for the first band only (we supposed this is the same for all the bands, 
            # and all bands have same number of rows columns)
            if return_index:
                index = bandarray[0]  # TODO this will overwrite the index for each band, should happens only once
                r_indexed.append(bandarray[1])
            else:
                r_indexed.append(bandarray) # TODO improve memory usage
            return_index = False

            # update the properties for this band
            # TODO: store only once
            properties["bands"]["columns"].append(band.XSize)
            properties["bands"]["rows"].append(band.YSize)
            properties["bands"]["blocksize"].append(band.GetBlockSize())
            properties["bands"]["nodatavalue"].append(band.GetNoDataValue())

            band = None  # give memory control back to C++ code

        #  group the indexed arrays
        if bandLocation == "bycolumn":
            #  create a 2darray, each row is a pixel, each column is a band
            r_indexed = np.vstack(r_indexed).T
        else:
            # create a 2darray, each row is a band, each column is a pixel
            r_indexed = np.vstack(r_indexed)

        return index, r_indexed, properties

    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e


def raster_list_to_indexed_numpy(dataset, name, bandLocation="bycolumn", nodata=-3.40282e+038):
    """Convert a list of gdal datasets (single band) into in indexed array
        we get the index for the first band only (we supposed this is the same for all the bands, 
        and all bands MUST HAVE the same number of rows and columns)

    :param dataset: list of raster datasets
    :param name: the name
    :param bandLocation: how to store bands (possible values: "byrow" , "bycolumn")
    :param nodata: pixel no data value
    :return: the index, the 2d indexed array , dictionary with raster properties (the properties of the first raster dataset in the list)
    """
    
    if len(dataset) <2 :
        raise Exception("number of bands should be >=2!!!")
    if not (bandLocation == "byrow" or bandLocation == "bycolumn"):
        raise Exception("you need the specify the correct \"bandLocation\" argument!!!")
    try:
        # describe the first dataset and set a dictionary with the raster properties
        name, width, height, nbands, csystem, geotransform = describe_raster_dataset(dataset[0], name)
        properties = {"name": name, "columns": width, "rows":height, "nbands": nbands, "csystem": csystem, 
                      "geotransform": geotransform, "bands orientation": bandLocation, "bands": {}}
        properties["bands"] = {"columns": [], "rows": [], "blocksize": [], "nodatavalue": []}

        # get the number of rasters
        nrasters=len(dataset)
        print("This list has " + str(nrasters) + " datasets")
        # update the number of bands
        properties["nbands"] = nrasters

        print("Preparing to process rasters....")

        # initialize a list of indexed arrays
        r_indexed=list()

        # iterate the bands and get an index for the first band and the indexed arrays
        # we get the index for the first band only (we supposed this is the same for all the bands, 
        # and all bands have same number of rows columns)
        
        print("We are processing "+ str(nrasters) + " bands")
        return_index = True
        for idx in range(0, nrasters):
            band = dataset[idx].GetRasterBand(1) #gdal bans start at 1
            if band is None:
                continue
            print("processing band " + str(idx+1))
            bandarray = single_band_to_indexed_numpy(band, nodata, return_index)
            
            # we get the index for the first band only (we supposed this is the same for all the bands, 
            # and all bands have same number of rows columns)
            
            if return_index:
                index = bandarray[0]  # TODO this will overwrite the index for each band, should happens only once
                r_indexed.append(bandarray[1])
            else:
                r_indexed.append(bandarray)
            return_index = False

            # update the properties for this band
            properties["bands"]["columns"].append(band.XSize)
            properties["bands"]["rows"].append(band.YSize)
            properties["bands"]["blocksize"].append(band.GetBlockSize())
            properties["bands"]["nodatavalue"].append(band.GetNoDataValue())

            band = None # give memory control back to C++ code

        # group the indexed arrays
        if bandLocation == "bycolumn":
            # create a 2darray, each row is a pixel, each column is a band
            r_indexed = np.vstack(r_indexed).T
        else:
            # create a 2darray, each row is a band, each column is a pixel
            r_indexed = np.vstack(r_indexed)

        return index, r_indexed, properties

    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e


def indexed_numpy_to_raster_dataset(data, outname, outfolder=".", datatype=gdct.GDT_Float32, nodata=-3.40282e+038, returnlist=True, frmt="GTiff"):
    
    """Create a raster dataset from an indexed numpy array and save it to disk
    
    :param data: input tuple (index+indexed array+properties dictionary)
    :param outname: what name?
    :param outfolder: where do you want to save?
    :param datatype: gdal raster datatype
    :param nodata: nodata value
    :param returnlist: return a list with the bands?
    :param frmt: gdal raster format?
    :return: 
    """

    try:
        if data[1].ndim == 1:
            nbands = 1
        else:
            # get the number of bands
            if data[2]["bands orientation"] == "bycolumn":
                nbands = data[1].shape[1]
            else: 
                nbands = data[1].shape[0]

        print("This array has "+str(nbands)+" bands")
        # create a list to store the bands
        if returnlist:
            bands=[]

        print("initializing new raster...")

        # set correct driver
        driver = gdal.GetDriverByName(frmt)
        driver.Register()

        # define the output dataset with the correct projection and geotransform
        outraster = driver.Create(outfolder+"/"+outname, data[2]["columns"], data[2]["rows"], nbands,  datatype)
        outraster.SetProjection(data[2]["csystem"])
        outraster.SetGeoTransform(data[2]["geotransform"])

        # create band by band
        print("getting band data....")
        for i in range(nbands):

            # set an empty numpy array for the band
            # a = np.zeros((data[2]["rows"]*data[2]["columns"])) + nodata
            a = np.zeros((data[2]["rows"],data[2]["columns"])) + nodata  # TODO: here set the correct datatype
            # fill np array with data from the indexed np array using the index

            if nbands > 1:
                if data[2]["bands orientation"] == "bycolumn":
                    a[data[0]] = data[1][:, i].ravel()
                else:  # byrows
                    a[data[0]] = data[1][i, :].ravel()
            else:  # single band raster
                # a[data[0]] = data[1][:].flatten()
                a[data[0]] = data[1][:].ravel()
            # print (a)
            # print (a.shape[0])

            if returnlist:
                bands.append(a)

            # write the band data
            outband = outraster.GetRasterBand(i+1)  # i+1 because gdal bands stars at 1
            outband.SetNoDataValue(nodata)
            outband.WriteArray(a)
            outband.FlushCache()
            print("band", str(i+1), "added")

        outband = None
        outraster = None

        if returnlist:
            return bands

    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e


def indexed_numpy_to_list(data, nodata = -3.40282e+038):  
    """ Create a a list where each element as a 1dimensional array for each raster band
    :param data: input tuple (index+indexed array+properties dictionary)
    :param nodata: pixel nodata value
    :return: a list of bands 
    """

    try:

        if data[1].ndim == 1:
            nbands = 1
        else:
            # get the number of bands
            if data[2]["bands orientation"] == "bycolumn":
                nbands = data[1].shape[1]
            else:
                nbands = data[1].shape[0]

        print("This array has " + str(nbands) + " bands")
        # create a list to store the bands
        bands = []

        # create band by band
        print("getting band data....")
        for i in range(nbands):

            # set an empty numpy array for the band
            # a = np.zeros((data[2]["rows"]*data[2]["columns"])) + nodata
            a = np.zeros((data[2]["rows"],data[2]["columns"])) + nodata  # TODO: here set the correct datatype
            # fill np array with data from the indexed np array using the index

            if nbands > 1:
                if data[2]["bands orientation"] == "bycolumn":
                    a[data[0]] = data[1][:,i].ravel()
                else:  # byrows
                    a[data[0]] = data[1][i,:].ravel()
            else:  # single band raster
                # a[data[0]] = data[1][:].flatten()
                a[data[0]] = data[1][:].ravel()
            # print (a)
            # print (a.shape[0])

            bands.append(a)
            print("band", str(i+1), "added")

        return bands

    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e


###### testing functions
if __name__ == "__main__":

    import os
    from osgeo import gdal
    import fileIO

    # test for raster properties
    def describeRaster():
        os.chdir(r"C:\xxx")
        rasterpath = r"new\EV_2003_al_warp.bsq"
        d = None
        try:
            d = gdal.Open(rasterpath)
            return describe_raster_dataset(d, os.path.basename(rasterpath))

        except Exception as e:
            print(e)
        finally:
            if d is not None:
                d = None  # give memory control back to C++ code

    # test for 1 band
    def importSingleBand():
        d = None
        band = None
        try:
            os.chdir(r"C:\xxx")
            rasterpath = r"new\EV_2003_al_warp.bsq"

            d = gdal.Open(rasterpath)
            band = d.GetRasterBand(1)
            print ("exporting to indexed numpy array")
            a = single_band_to_indexed_numpy(band)
            print ("saving object to disk")
            fileIO.save_object("C:/xxx/EV_2003_single", a)
            print ("loading object from disk")
            c=fileIO.load_object("C:/xxx/EV_2003_single")
            return a,c

        except Exception as e:
            print (e)
        finally:
            if d is not None:
                d = None
            if band is not None:
                band = None

    # test for multiband
    def importRaster():
        d = None
        try:
            os.chdir(r"C:\xxx")
            rasterpath = r"new\EV_2003_al_warp.bsq"
            d = gdal.Open(rasterpath)
            b = raster_dataset_to_indexed_numpy(d, os.path.basename(rasterpath), maxbands = 10, bandLocation="bycolumn", nodata= -3.40282e+038)
            print("saving object to disk")
            fileIO.save_object("C:/xxx/EV_2003_al_warp", b)
            print("loading object from disk")
            c = fileIO.load_object("C:/xxx/EV_2003_al_warp")
            return b, c

        except Exception as e:
            print(e)
        finally:
            if d is not None:
                d = None

    #test for multiband
    def importSingleBandRaster():
        d = None
        try:
            os.chdir(r"C:/xxx")
            rasterpath = r"NDVI2014-09-17.tif"
            d = gdal.Open(rasterpath)
            b = raster_dataset_to_indexed_numpy(d, os.path.basename(rasterpath), maxbands=1, bandLocation="bycolumn", nodata=-10)
            print("saving object to disk")
            fileIO.save_object("C:/xxx/NDVI2014-09-17", b)
            print("loading object from disk")
            c = fileIO.load_object("C:/xxx/NDVI2014-09-17")
            return b, c

        except Exception as e:
            print(e)
        finally:
            if d is not None:
                d = None

    # test for exporting indexed array
    def exportRaster():
        try:
            os.chdir(r"C:/xxx")
            print("loading object from disk")
            c = fileIO.load_object("NDVI2014-09-17")
            dataset = indexed_numpy_to_raster_dataset(c, "NDVI2014-09-17COPY.tif", outfolder=os.getcwd(), datatype=gdct.GDT_Float32, nodata=-10, returnlist=True, frmt="GTiff")
            return dataset

        except Exception as e:
            print(e)

    # test for list of datasets
    def importBandList():

        ls = []
        try:
            os.chdir(r"C:/xxx")
            rasterpath = r"NDVI2014-09-17.tif"

            for i in range(4):
                ls.append(gdal.Open(rasterpath))

            return raster_list_to_indexed_numpy(ls, "lista", bandLocation="bycolumn", nodata=-3.40282e+038)
        finally:
            for i in ls:
                i = None

    # descr=describeRaster(); print(descr)
    # a,c =importSingleBand(); print(a, c)
    # multi=importRaster();print(multi[0], multi[1])
    # single2=importSingleBandRaster(); print(single2[0], single2[1])
    # raster=exportRaster(); print(raster)
    # index, rIndexed, properties = importBandList(); print(index, rIndexed, properties)
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:        utils.py
# Purpose:      some spatial utility functions:
#                   -   get a gdal/ogr dataset spatial reference name
#                   -   get the spatial reference in different formats from a GDAL/OGR dataset
#                   -   reproject vectors
#                   -   reproject raster
#                   -   reproject raster (olf function nont working with geographical references)
#                   -   change coordinates from projected to geographic and viceversa for pandas dataframe columns
#                   -   save ogr dataset to disk
#                   -   save gdal dataset to disk
#                   -   to get envelope coordinates
#                   -   to compare attribute tables
#                   -   to get the first geometry from a shapefile.
#                   -   Determine if the current GDAL is built with SpatiaLite support
#                   -   Print attribute values in a layer
#                   -   Print capabilities for a driver, datasource, or layer
#                   -   Print a list of available drivers
#                   -   Print a list of layers in a data source
#                   -   Get a geometry string for printing attributes ("private" function)
#                   -   Get attribute values from a feature ("private" function)
#                   -   Get the datasource and layer from a filename ("private" function)
#                   -   Print capabilities for a driver, datasource, or layer. ("private" function)
#                   -   Convert/Rescale image (need orfeo installed)
#                   -   Get min max values from an image band
#                   -   Convert a csv file to a polyline shapefile
#                   -   Buffer a shapefile
#                   -   Read the values of raster cells under a vector point point shapefile
#                   -   extracting raster values using csv points
#                   -   generate point wkt from x,y coordinates
#                   -   Format a gdal vrt xml file and save it to disk
#                   -   Assign row number to ordered field points
#                   -   Reclassify a raster given a threshold
#                   -   Get the EPSG UTM WGS84 code given the latitude and longitude
#
# Author:      Claudio Piccinini, some functions based on Chris Garrard code
#
# Created:     13/03/2015, updated august 2017
# -------------------------------------------------------------------------------

import sys
import math
import codecs
import os

import numpy as np
import pandas as pd


import osgeo #this is necessary for the type comparison in some methods
from osgeo import osr
from osgeo import gdal
from osgeo import gdal_array as gdar
from osgeo import gdalconst as gdct
from osgeo import ogr
gdal.UseExceptions()

from pyproj import Proj

_geom_constants = {}
_ignore = ["wkb25DBit", "wkb25Bit", "wkbXDR", "wkbNDR"]
for c in filter(lambda x: x.startswith("wkb"), dir(ogr)):
    if c not in _ignore:
        _geom_constants[ogr.__dict__[c]] = c[3:]


def get_coordinate_name(dataset, rawsrs=False):
    """ get a dataset spatial reference name
    :param gdal/ogr dataset:
    :param rawsrs: if true return the spatial reference object, false return a string with info
    :return: the name of the spatial reference
    """
    if isinstance(dataset, osgeo.gdal.Dataset):

        prj=dataset.GetProjection()
        srs=osr.SpatialReference(wkt=prj)
    else:
        layer = dataset.GetLayer()
        srs = layer.GetSpatialRef()
    if rawsrs:
        return srs
    else:
        if srs.IsProjected():
            return srs.GetAttrValue("projcs")
        else: return srs.GetAttrValue("geogcs")


def export_spatialref(dataset):
    """ get the spatial reference in different formats from a GDAL/OGR dataset
    :param dataset: a gdal/ogr raster dataset
    :return: a dictionary
    """
    if isinstance(dataset, osgeo.gdal.Dataset):
        spatialRef = osr.SpatialReference()
        wkt = dataset.GetProjection()
        spatialRef.ImportFromWkt(wkt)
    else:
        layer = dataset.GetLayer()
        spatialRef = layer.GetSpatialRef()

    print(spatialRef)
    out={}
    out["Wkt"] =spatialRef.ExportToWkt()
    out["PrettyWkt"] =spatialRef.ExportToPrettyWkt()
    out["PCI"] =spatialRef.ExportToPCI()
    out["USGS"] =spatialRef.ExportToUSGS()
    out["XML"] =spatialRef.ExportToXML()

    return out


def reproject_vector(inDataSet, epsg_from=None, epsg_to=None):
    
    """ reproject a vector file (only the first layer!) (it does not save the dataset to disk)
    :param inDataSet: the input ogr dataset
    :param epsg_from: the input spatial reference; in None take the source reference
    :param epsg_to: the output spatial reference
    :return: the reprojected dataset
    """

    if not epsg_to: raise Exception("please, specify the output EPSG codes")

    outDataSet = None
    inFeature = None
    outFeature = None
    outLayer = None

    try:
        #driver = inDataSet.GetDriver()

        # define input SpatialReference
        if not epsg_from:
            layer = inDataSet.GetLayer()
            inSpatialRef = layer.GetSpatialRef()
        else:
            inSpatialRef = osr.SpatialReference()
            inSpatialRef.ImportFromEPSG(epsg_from)

        # define output SpatialReference
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(epsg_to)

        # create the CoordinateTransformation
        coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

        # get the first input layer and the geometry type
        inLayer = inDataSet.GetLayer()
        geotype = inLayer.GetGeomType()
        lname = inLayer.GetName()

        drv = ogr.GetDriverByName("ESRI Shapefile")
        outDataSet = drv.CreateDataSource( "/vsimem/memory.shp" )

        outLayer = outDataSet.CreateLayer(lname, srs = outSpatialRef, geom_type = geotype)

        # add fields
        inLayerDefn = inLayer.GetLayerDefn()

        for i in range(0, inLayerDefn.GetFieldCount()):
            fieldDefn = inLayerDefn.GetFieldDefn(i)
            outLayer.CreateField(fieldDefn)

        # get the output layer"s feature definition
        outLayerDefn = outLayer.GetLayerDefn()


        counter=1

        # loop through the input features
        inFeature = inLayer.GetNextFeature()
        while inFeature:
            # get the input geometry
            geom = inFeature.GetGeometryRef()
            # reproject the geometry
            geom.Transform(coordTrans)
            # create a new feature
            outFeature = ogr.Feature(outLayerDefn)
            # set the geometry and attribute
            outFeature.SetGeometry(geom)
            for i in range(0, outLayerDefn.GetFieldCount()):
                outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
            # add the feature to the shapefile
            outLayer.CreateFeature(outFeature)

            # destroy the features and get the next input feature
            if outFeature: outFeature = None
            inFeature = inLayer.GetNextFeature()

            counter += 1
            print(counter)

        return outDataSet

    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e

    finally:
        if outDataSet: outDataSet == None #give back control to C++
        if outLayer: outLayer == None
        if inFeature: inFeature == None
        if outFeature: outFeature = None


def reproject_raster(dataset, epsg_from = None, epsg_to=None, fltr=gdal.GRA_NearestNeighbour):
    """reproject a gdal raster dataset
    :param dataset: a gdal dataset
    :param epsg_from: the input epsg; if None get from the sorce
    :param epsg_to: the output epsg; if None throw exception
    :param fltr: the filter to apply when reprojecting
        GRA_NearestNeighbour
        Nearest neighbour (select on one input pixel)
        GRA_Bilinear
        Bilinear (2x2 kernel)
        GRA_Cubic
        Cubic Convolution Approximation (4x4 kernel)
        GRA_CubicSpline
        Cubic B-Spline Approximation (4x4 kernel)
        GRA_Lanczos
        Lanczos windowed sinc interpolation (6x6 kernel)
        GRA_Average
        Average (computes the average of all non-NODATA contributing pixels)
        GRA_Mode
        Mode (selects the value which appears most often of all the sampled points)

    #############NearestNeighbour filter is good for categorical data###########

    :return: the reprojected dataset
    """

    try:

        if epsg_to is None:
            raise Exception("select the destination projected spatial reference!!!")

        if epsg_from == epsg_to:
            print("the input and output projections are the same!")
            return dataset

        # Define input/output spatial references
        if epsg_from:
            source = osr.SpatialReference()
            source.ImportFromEPSG(epsg_from)
            inwkt = source.ExportToWkt()
        else:
            source = osr.SpatialReference()
            source.ImportFromWkt(dataset.GetProjection())
            source.MorphFromESRI()  #this is to avoid reprojection errors
            inwkt = source.ExportToWkt()

        destination = osr.SpatialReference()
        destination.ImportFromEPSG(epsg_to)
        outwkt = destination.ExportToWkt()

        vrt_ds = gdal.AutoCreateWarpedVRT(dataset, inwkt, outwkt, fltr)

        return vrt_ds

    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e

################################################################################
# this is an old function to reproject rasters, it allows to define the output pixel size
# NOTE: it does not work with geographic spatial references


def reproject_raster_1(dataset, pixel_spacing = None, epsg_from=None, epsg_to=None, fltr=gdal.GRA_NearestNeighbour):
    """ reproject a raster. It does not work with geographic spatial references
    :param dataset: a gdal dataset
    :param pixel_spacing: the output pixel width and pixel height (we assume they are the same), if None use source
    :param epsg_from: the input epsg; if None get from the sorce
    :param epsg_to: the output epsg; if None throw exception
    :param fltr: the filter to apply when reprojecting
                GRA_NearestNeighbour
                Nearest neighbour (select on one input pixel)
                GRA_Bilinear
                Bilinear (2x2 kernel)
                GRA_Cubic
                Cubic Convolution Approximation (4x4 kernel)
                GRA_CubicSpline
                Cubic B-Spline Approximation (4x4 kernel)
                GRA_Lanczos
                Lanczos windowed sinc interpolation (6x6 kernel)
                GRA_Average
                Average (computes the average of all non-NODATA contributing pixels)
                GRA_Mode
                Mode (selects the value which appears most often of all the sampled points)

    :return: the reprojected dataset

    #############NearestNeighbour filter is good for categorical data###########

    """

    mem_drv = None

    try:

        if epsg_to is None:
            raise Exception("select the destination projected spatial reference!!!")

        # Define spatial references and transformation
        source = osr.SpatialReference()
        if epsg_from:
            source.ImportFromEPSG(epsg_from)
        else:
            wkt = dataset.GetProjection()
            source.ImportFromWkt(wkt)
            source.MorphFromESRI()  # this is to avoid reprojection errors

        destination = osr.SpatialReference()
        destination.ImportFromEPSG(epsg_to)

        # print(source.GetAttrValue("projcs"))
        # print(source.GetAttrValue("geogcs"))
        # check we have projected spatial references
        if destination.IsGeographic() or source.IsGeographic():
            raise Exception("geographic spatial reference are not allowed (still...)")

        tx = osr.CoordinateTransformation(source, destination)

        # get the number of bands
        nbands = dataset.RasterCount

        # get the data type from the first band
        go = True
        while go:
            band = dataset.GetRasterBand(1)
            if band is None: continue
            go = False
            btype = band.DataType
            if band is not None: band = None #give back control to C++

        # Get the Geotransform vector
        geo_t = dataset.GetGeoTransform ()
        x_size = dataset.RasterXSize  # Raster xsize
        y_size = dataset.RasterYSize  # Raster ysize

        # Work out the boundaries of the new dataset in the target projection
        # TODO what if the input raster is rotated?
        (ulx, uly, ulz) = tx.TransformPoint(geo_t[0], geo_t[3])
        (llx, lly, llz) = tx.TransformPoint(geo_t[0], geo_t[3] + geo_t[5]*y_size)

        (lrx, lry, lrz) = tx.TransformPoint(geo_t[0] + geo_t[1]*x_size, geo_t[3] + geo_t[5]*y_size )
        (urx, ury, urz) = tx.TransformPoint(geo_t[0] + geo_t[1]*x_size, geo_t[3])

        ulx = min(ulx, llx)
        uly = max(uly, ury)  #TODO check for the southern emisphere

        lrx = max(lrx,urx)
        lry = min(lry,lly)  #TODO check for the southern emisphere


        # set the input spacing if user did not set it
        if pixel_spacing is None:
            pixel_spacing = geo_t[1]

        # Now, we create an in-memory raster
        mem_drv = gdal.GetDriverByName("MEM")
        dest = mem_drv.Create("", int((lrx - ulx)/pixel_spacing), int((uly - lry)/pixel_spacing), nbands, btype)

        # Calculate the new geotransform
        new_geo = ( ulx, pixel_spacing, geo_t[2], uly, geo_t[4], -pixel_spacing) #TODO the - might not work in the southern emisphere

        # Set the geotransform
        dest.SetGeoTransform(new_geo)
        dest.SetProjection(destination.ExportToWkt())

        # Perform the projection/resampling
        res = gdal.ReprojectImage(dataset, dest, source.ExportToWkt(), destination.ExportToWkt(), fltr)

        return dest

    except RuntimeError as err:
        raise err
    except Exception as e:
         raise e

    finally:
        if mem_drv is not None:
            mem_drv == None  # give back control to C++


def reproject_coordinates(df, projstring="epsg:32611", x="longitude", y="latitude", inverse=False):
    """
    Change coordinates from projected to geographic and viceversa for pandas dataframe columns
    :param df: pandas dataframe
    :param  projstring: projection string
    :param x: column name
    :param y:column name
    :param inverse: True to pass from projected to geographic
    :return: the 2 columns as numpy arrays
    """
    p = Proj(init=projstring)
    w, n = p(df[x].values, df[y].values, inverse=inverse)
    return w, n


def save_vector(dataset, outpath, driver=None):
    """ save an ogr dataset to disk, (it will delete preexisting output)
    :param dataset: ogr dataset
    :param outpath: output path
    :param driver: override with a driver name, otherwise the driver will be inferred from the dataset
    :return: None
    """
    try:
        if not driver:
            driver = dataset.GetDriver()
            if os.path.exists(outpath):
                driver.DeleteDataSource(outpath)
            dst_ds = driver.CopyDataSource(dataset, outpath)
        else:
            driver = ogr.GetDriverByName(driver)
            if os.path.exists(outpath):
                driver.DeleteDataSource(outpath)
            dst_ds = driver.CopyDataSource(dataset, outpath)


    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e

    finally:
        dst_ds = None  # Flush the dataset to disk


def save_raster(dataset, outpath, driver=None):

    """save an gdal dataset to disk, (it will delete preexisting output)
    :param dataset: gdal dataset
    :param outpath: output path
    :param driver: override with a driver name, otherwise the driver will be inferred from the dataset
    :return: None
    """

    try:

        if not driver:
            driver = dataset.GetDriver()
            if os.path.exists(outpath):
                driver.Delete(outpath)
            dst_ds = driver.CreateCopy(outpath, dataset)
        else:  #TODO: test if this works
            driver = gdal.GetDriverByName(driver)
            if os.path.exists(outpath):
                driver.Delete(outpath)
            dst_ds = driver.CreateCopy(outpath, dataset)

    except RuntimeError as err:
        raise err
    except Exception as e:
         raise e

    finally:
        dst_ds = None # Flush the dataset to disk


################################################################################

def get_envelope_coordinates(fc, start=0, stop=None, prnt=False):
    """get the envelopes coordinates of an ogr dataset; it's possible to select a range of features
    :param fc:the ogr dataset
    :param start: positive start index; first index is 0 ; last index is fc.featureCount()-1
    :param stop: positive stop index; stop index is not comprised in the result
    :param prnt: if True print the coordinate on screen
    :return: a list of coordinates (by default get all the features)
    """

    raise Exception("not implemented")


def compare_tables(at1,at2):
    """ compare 2 attribute tables
    :param at1: attribute table 1
    :param at2: attribute table 2
    :return: a list of cell differences
    """

    raise Exception("not implemented")



######################################################################


def get_shp_geom(fn):
    """ OGR layer object or filename to datasource (will use 1st layer)
    :param fn:
    :return:
    """

    lyr, ds = _get_layer(fn)
    feat = lyr.GetNextFeature()
    return feat.geometry().Clone()


def has_spatialite():
    """Determine if the current GDAL is built with SpatiaLite support."""
    use_exceptions = ogr.GetUseExceptions()
    ogr.UseExceptions()
    try:
        ds = ogr.GetDriverByName("Memory").CreateDataSource("memory")
        sql = '''SELECT sqlite_version(), spatialite_version()'''
        lyr = ds.ExecuteSQL(sql, dialect="SQLite")
        return True
    except Exception as e:
        return False
    finally:
        if not use_exceptions:
            ogr.DontUseExceptions()


def print_attributes(lyr_or_fn, n=None, fields=None, geom=True, reset=True):
    """ Print attribute values in a layer.
    :param lyr_or_fn: OGR layer object or filename to datasource (will use 1st layer)
    :param n: optional number of features to print; default is all
    :param fields: optional list of case-sensitive field names to print; default is all
    :param geom: optional boolean flag denoting whether geometry type is printed; default is True
    :param reset: optional boolean flag denoting whether the layer should be reset to the first record before printing; default is True
    :return:
    """

    lyr, ds = _get_layer(lyr_or_fn)
    if reset:
        lyr.ResetReading()

    n = n or lyr.GetFeatureCount()
    geom = geom and lyr.GetGeomType() != ogr.wkbNone
    fields = fields or [field.name for field in lyr.schema]
    data = [["FID"] + fields]
    if geom:
        data[0].insert(1, "Geometry")
    feat = lyr.GetNextFeature()
    while feat and len(data) <= n:
        data.append(_get_atts(feat, fields, geom))
        feat = lyr.GetNextFeature()
    lens = map(lambda i: max(map(lambda j: len(str(j)), i)), zip(*data))
    format_str = "".join(map(lambda x: "{{:<{}}}".format(x + 4), lens))
    for row in data:
        try:
            print(format_str.format(*row))
        except UnicodeEncodeError:
            e = sys.stdout.encoding
            print(codecs.decode(format_str.format(*row).encode(e, "replace"), e))
    print("{0} of {1} features".format(min(n, lyr.GetFeatureCount()), lyr.GetFeatureCount()))
    if reset:
        lyr.ResetReading()


def print_capabilities(item):
    """Print capabilities for a driver, datasource, or layer."""
    if isinstance(item, ogr.Driver):
        _print_capabilites(item, "Driver", "ODrC")
    elif isinstance(item, ogr.DataSource):
        _print_capabilites(item, "DataSource", "ODsC")
    elif isinstance(item, ogr.Layer):
        _print_capabilites(item, "Layer", "OLC")
    else:
        print("Unsupported item")


def print_drivers():
    """Print a list of available drivers."""
    for i in range(ogr.GetDriverCount()):
        driver = ogr.GetDriver(i)
        writeable = driver.TestCapability(ogr.ODrCCreateDataSource)
        print("{0} ({1})".format(driver.GetName(),
                                 "read/write" if writeable else "readonly"))


def print_layers(fn):
    """ Print a list of layers in a data source.
    :param fn: path to data source
    :return:
    """

    ds = ogr.Open(fn, 0)
    if ds is None:
        raise OSError("Could not open {}".format(fn))
    for i in range(ds.GetLayerCount()):
        lyr = ds.GetLayer(i)
        print("{0}: {1} ({2})".format(i, lyr.GetName(), _geom_constants[lyr.GetGeomType()]))


def _geom_str(geom):
    """ Get a geometry string for printing attributes.
    :param geom:  gdal geometry
    :return: geometry name
    """
    if geom.GetGeometryType() == ogr.wkbPoint:
        return "POINT ({:.3f}, {:.3f})".format(geom.GetX(), geom.GetY())
    else:
        return geom.GetGeometryName()


def _get_atts(feature, fields, geom):
    """Get attribute values from a feature.
    :param feature: input feature
    :param fields: which fields you want to get?
    :param geom: do you want the geometry?
    :return: a list with attributes
    """
    data = [feature.GetFID()]
    geometry = feature.geometry()
    if geom and geometry:
        data.append(_geom_str(geometry))
    values = feature.items()
    data += [values[field] for field in fields]
    return data


def _get_layer(lyr_or_fn):
    """ Get the datasource and layer from a filename.
    :param lyr_or_fn: filename
    :return: layer and dataset
    """

    if type(lyr_or_fn) is str:
        ds = ogr.Open(lyr_or_fn)
        if ds is None:
            raise OSError("Could not open {0}.".format(lyr_or_fn))
        return ds.GetLayer(), ds
    else:
        return lyr_or_fn, None


def _print_capabilites(item, name, prefix):
    """ Print capabilities for a driver, datasource, or layer.
    :param item: item to test
    :param name: name of the type of item
    :param prefix: prefix of the ogr constants to use for testing
    :return: None
    """
    print("*** {0} Capabilities ***".format(name))
    for c in filter(lambda x: x.startswith(prefix), dir(ogr)):
        print("{0}: {1}".format(c, item.TestCapability(ogr.__dict__[c])))


def convert_raster(imgpath=None, outpath=None, outdatatype="uint8", rescaletype="linear", exepath="otbcli_Convert.bat"):
    """ convert a raster to a new pixel format
    :param imgpath: input raster
    :param outpath: output raster
    :param outdatatype: output data type
    :param rescaletype: conversion type
    :param exepath: the path to conversion utility (change if the system path variable is not set)
    :return: messages
    """
    import subprocess
    params = [exepath, "-in", imgpath, "-out", outpath, outdatatype, "-type", rescaletype]
    p = subprocess.Popen(params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # A pipe is a section of shared memory that processes use for communication.
    out, err = p.communicate()
    return bytes.decode(out),bytes.decode(err)


def get_minmax(imagepath, band = 1 ):
    """ get the minimum and maximum of a raster band
    :param   input raster path
    :param   band number to analyse (starts at 1)
    """
    d = None

    try:
        d = gdal.Open(imagepath)
        band = d.GetRasterBand(band) 
        band.ComputeStatistics(False)
        mn = band.GetMinimum()
        mx = band.GetMaximum()
        return mn, mx
    except Exception as e:
        print(e)
    finally:
        if d:
            d = None


def csv_to_polyline_shapefile(df, ycol="lat", xcol="lon", linecol="row", epsg=4326, outpath = "output.shp", fastMethod = True):
    """
    Convert a csv file to a polyline shapefile
    code based on https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html#create-a-new-shapefile-and-add-data

    :param df: path to the csv file or pandas dataframe
    :param ycol: field with y coordinates
    :param xcol: field with x coordinates
    :param linecol: integer field used to aggregate points to the same line
    :param epsg: epsg code to aasign to the output (data is not reprojected)
    :param outpath: output path
    :param fastMethod: True to consider only the first and last point of the row
    :return:
    """

    data_source = None

    try:

        # check the input is a csv or a pandas datframe
        if type(df) == str:
            df = pd.read_csv(df, usecols=[linecol,ycol,xcol])
        elif type(df) == pd.core.frame.DataFrame:
            pass
        else:
            raise TypeError ('pass a csv or a pandas dataframe')

        # set up the shapefile driver
        driver = ogr.GetDriverByName("ESRI Shapefile")

        # create the data source
        if os.path.exists(outpath):
            driver.DeleteDataSource(outpath)
        data_source = driver.CreateDataSource(outpath)

        # create the spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg))

        # create the layer
        layer = data_source.CreateLayer(os.path.basename(outpath).split(".")[0], srs, ogr.wkbLineString)

        # Add the fields we're interested in
        layer.CreateField(ogr.FieldDefn("id_row", ogr.OFTInteger))

        #get the unique row id
        rows = df[linecol].unique()

        # subset using the row id and build the wkt
        for row in rows:

            print("Adding line" + str(row) )
            # select one vineyard row
            fdf = df[df[linecol] == row]

            # create the feature
            feature = ogr.Feature(layer.GetLayerDefn())
            # Set the attributes using the values from the delimited text file
            feature.SetField("id_row",str(row) )

            wkt = []

            if fastMethod: #use only start and end point
                wkt.append(str(fdf.head(1)[xcol].values[0]) + " " + str(fdf.head(1)[ycol].values[0]))
                wkt.append(str(fdf.tail(1)[xcol].values[0]) + " " + str(str(fdf.tail(1)[ycol].values[0])))
                wkt = "LINESTRING (" + (",".join(wkt)) + ")"
                # print(wkt)
            else: #use all the points
                #iterate the points and create the wkt string
                for index, therow in fdf.iterrows():  #todo find a more efficient way to create a line, maybe pyshp library
                # create the WKT for the feature using Python string formatting
                    wkt.append(str(therow[xcol])+ " " +str(therow[ycol]))
                wkt = "LINESTRING ("+ (",".join(wkt)) + ")"
                #print(wkt)

            # Create the line from the Well Known Txt
            line = ogr.CreateGeometryFromWkt(wkt)
            # Set the feature geometry using the point
            feature.SetGeometry(line)
            # Create the feature in the layer (shapefile)
            layer.CreateFeature(feature)
            # Dereference the feature
            feature = None

    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e

    finally:
        # Save and close the data source
        if data_source: data_source = None


def buffer(inputfn, outputBufferfn, bufferDist):
    """Buffer a shapefile
    code based on http://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html?highlight=buffer#create-buffer
    :param inputfn:  path to the input
    :param outputBufferfn: path to the output
    :param bufferDist: buffer distance
    :return:
    """
    inputds = None
    outputBufferds = None

    try:

        shpdriver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(outputBufferfn):
            shpdriver.DeleteDataSource(outputBufferfn)
        outputBufferds = shpdriver.CreateDataSource(outputBufferfn)

        inputds = ogr.Open(inputfn)
        inputlyr = inputds.GetLayer()

        # if a coordinate system is available copy it
        srs = get_coordinate_name(inputds, rawsrs=True)
        if srs:
            bufferlyr = outputBufferds.CreateLayer(os.path.basename(outputBufferfn).split(".")[0],srs,geom_type=ogr.wkbPolygon)
        else:
            bufferlyr = outputBufferds.CreateLayer(os.path.basename(outputBufferfn).split(".")[0],geom_type=ogr.wkbPolygon)

        # set the output layer fields
        featureDefn = inputlyr.GetLayerDefn()
        for i in range(0, featureDefn.GetFieldCount()):
            fieldDefn = featureDefn.GetFieldDefn(i)
            bufferlyr.CreateField(fieldDefn)
        outLayerDefn = bufferlyr.GetLayerDefn()

        for feature in inputlyr:

            print("buffering feature" + str(feature))

            ingeom = feature.GetGeometryRef()
            geomBuffer = ingeom.Buffer(bufferDist)
            outFeature = ogr.Feature(outLayerDefn)
            outFeature.SetGeometry(geomBuffer)
            for i in range(0, outLayerDefn.GetFieldCount()):
                outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), feature.GetField(i))
            bufferlyr.CreateFeature(outFeature)
            outFeature = None

    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e
    finally:
        # Save and close the data source
        if inputds: inputds = None
        if outputBufferds: outputBufferds = None


def read_point_pixel_coords(shp_filename, raster_filename):
    """
    Read the coordinates for a point shapefile over a raster
    Only works for rasters with no rotation.
    :param shp_filename: shapefile
    :param raster_filename: raster
    :return: a list with pixel values for each point
    """

    src_ds = None
    ds = None

    try:

        # Todo check this is a point geometry!
        ds = ogr.Open(shp_filename)
        lyr = ds.GetLayer()


        #print(raster_filename)
        src_ds = gdal.Open(raster_filename)
        gt = src_ds.GetGeoTransform()
        rb = src_ds.GetRasterBand(1)

        pixels = []

        for feat in lyr:
            geom = feat.GetGeometryRef()
            mx, my = geom.GetX(), geom.GetY()  # coord in map units

            # Convert from map to pixel coordinates.
            # Only works for geotransforms with no rotation.
            px = int((mx - gt[0]) / gt[1])  # x pixel
            py = int((my - gt[3]) / gt[5])  # y pixel

            intval = rb.ReadAsArray(px, py, 1, 1)
            pixels.append(intval[0][0])  # intval is a numpy array, length=1 as we only asked for 1 pixel value

        return pixels

    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e
    finally:
        if src_ds: src_ds = None
        if ds: ds = None


def compare_csvpoint_cell(point, raster, vrtdict, overwrite=True, **kwargs):
    """
    extracting raster values using csv points
    :param point: path to csv file with points (must have the header!)
    :param raster: raster to query
    :param vrtdict: dictionary with values for vrt file generation
    :param overwrite: True to overwriteexisting shapefile

    vrtdict == {"layername": , "fullname": , "easting": , "northing":  , "srs"}

    :return: return a list with the raster values
    """

    # create a virtual data sourse and save to disk
    xml = '''<OGRVRTDataSource>
                <OGRVRTLayer name="{layername}">
                    <SrcDataSource>{fullname}</SrcDataSource>
                    <GeometryType>wkbPoint</GeometryType>
                    <LayerSRS >{srs}</LayerSRS >
                    <GeometryField encoding="PointFromColumns" x="{easting}" y="{northing}" />
                </OGRVRTLayer>
    </OGRVRTDataSource>
    '''

    in_ds = None

    try:

        folder = os.path.dirname(point)
        name = os.path.basename(point).split(".")[0]

        vrt = folder + "/" + name + ".vrt"  # the .vrt is saved in the same folder a the csv
        with open( vrt, "w") as f:
            newxml = xml.format(**vrtdict)
            #f = open( vrt, "w")
            f.write(newxml)
            #f.close()

        # converst csv to shapefile
        print("converting csv to shapefile")
        in_ds = ogr.Open(vrt)

        if overwrite:
            save_vector(in_ds,folder + "/" + name + ".shp", "ESRI Shapefile")
        else:
            if not os.path.exists(folder + "/" + name + ".shp"):
                save_vector(in_ds, folder + "/" + name + ".shp", "ESRI Shapefile")

        #print(folder + "/" + name + ".shp")

        #query the raster with the points
        print("Querying raster, this may take a while depending on the number of points")
        #return point_query(folder + "/" + name +  ".shp", raster)
        return  read_point_pixel_coords(folder + "/" + name + ".shp", raster)


    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e
    finally:

        if in_ds: in_ds = None


def genPointText(x,y):
    """ Generate the point WKT
    :param x: x coord
    :param y: y coord
    :return:  the WKT
    """
    generatedPoint = "POINT(%s %s)" % (x,y)
    return generatedPoint


def make_vrt(xml, data, outname):
    """
    Format a vrt file and save it to disk
    :param xml: format string for vrt generation

    for instance
    <OGRVRTDataSource>
                <OGRVRTLayer name="{layername}">
                    <SrcDataSource>{fullname}</SrcDataSource>
                    <GeometryType>wkbPoint</GeometryType>
                    <GeometryField encoding="PointFromColumns" x="{easting}" y="{northing}" z="{elevation}"/>
                </OGRVRTLayer>
    </OGRVRTDataSource>

    :param data: dictionary with values to assign to the xml

    data = {"layername": ""  "fullname": "", "easting": "", "northing": "", ......}

    :param outname: the path to save the vrt
    :return: None
    """
    newxml = xml.format(**data)
    with open(outname, "w") as f:
        f.write(newxml)


# todo the line number should be uploaded to the database
def point_to_numbered_line(file, xyfield, threshold=2):
    """
    Assign row number to ordered field points

    The threshold is used to separate points belonging to different lines,
    when row Direction is along x the threshold is applied to y coordinates
    when row Direction is along y the threshold is applied to x coordinates

    :param file: the file with the points or a pandas dataframe
    :param xyfield: the field name with the x/y coordinates
            When the rows are west-east these are the y coordiantes
            When the rows are north-south these are the x coordinates(y for rowDirection=="x")   
    :param threshold: threshold in coordinate system units 
    :return: an array with the row numbers
    """

    # 1 open the csv file to pandas
    # 2 extract the x or y fields
    # check the input is a csv or a pandas datframe (it's up to the user to pass the correct dataframe)

    if type(file) == str:
        df = pd.read_csv(file, usecols=[xyfield])
    elif type(file) == pd.core.frame.DataFrame:
        pass
    else:
        raise TypeError('pass a csv or a pandas dataframe')

    # 3 initialize an empty array with 4 columns
    arr = np.zeros((df.shape[0], 4))

    # 4 if rowDirection is x copy the y cordinates to the first column, else copy x coordinates
    arr[:, 0] = df[xyfield].values
    #  the y cordinates shifted by -1 in the second column
    arr[:-1, 1] = arr[1:, 0]

    # 5 copy the last value of the first column to the second column
    ####arr[-1, 1] = arr[-1, 0]

    # 6 calculate the difference between the 2 columns abd put the absolute value
    #   in column 3, also add a value larger than threshold at the end of the
    # column 3 to signal the end of the last row
    arr[:, 2] = np.abs(arr[:, 0] - arr[:, 1])
    arr[-1, 2] = threshold + 1
    # use the threshold to get the index of rows where the third column exceed the thresold
    index = np.nonzero(arr[:, 2] > threshold)

    # print(index[0])

    if index[0].size == 1:  # if there is only one row set row number to 1 for all data and return column
        arr[:, 3] = 1
        return arr[:, 3]

    # calculate the row number starting from 1 and put in the 4th column
    # first set the first row
    arr[0:index[0][0] + 1, 3] = 1  # index is a tuple  (indexes,) that's why the double index
    for x in range(index[0].size - 1):  # iterate all the indexes minus the last one
        start = index[0][x]
        end = index[0][x + 1]
        # the index signal the last point of a row, that's why the start+1
        # the end+1 because of python rules ( the last is not comprised)
        # the x+2 because we start with row 2
        arr[start + 1:end + 1, 3] = x + 2

    # print(arr[:, 3])

    # return the 4th column with the row numbers
    return arr[:, 3]


def reclassify_raster(infile, outfile, operator, threshold, pixel_value=100, frmt="GTiff", silence=True):
    """
    Reclassify a raster into a single value according to a threshold
    :param infile: path to the input
    :param outfile: path to the output
    :param operator: string, possible values ">","<",">=","<="
    :param threshold: threshold value
    :param pixel_value: the output pixel value for pixels that pass the threshold
    :param frmt: the output format, default "GTiff"
    :param silence: pass any value to enable debug info
    :return: None
    """

    '''using this instead of gdal_calc because gdal calc didn't work
    scriptpath = os.path.join(os.path.dirname(__file__), '../', "pysdss/utility/gdal_calc.py")
    scriptpath = os.path.normpath(scriptpath)
    params = [scriptpath,"-A",path+input,"--outfile="+path+output,"--calc='(A>0.5)'", "--debug"]
    print("Reclassify raster")
    out,err = utils.run_script(params, callpy=["python3"])
    print("output" + out)
    if err:
        print("ERROR:" + err)
        raise Exception("gdal grid failed")'''

    if operator not in [">","<",">=","<="]: raise Exception("unknown operator")

    band1 = None
    ds1 = None
    bandOut = None
    dsOut = None

    try:
        if not silence:
            print("opening input")

        # Open the dataset
        ds1 = gdal.Open(infile, gdct.GA_ReadOnly)
        band1 = ds1.GetRasterBand(1)
        # Read the data into numpy arrays
        data1 = gdar.BandReadAsArray(band1)
        # get the nodata value
        nodata = band1.GetNoDataValue()

        if not silence:
            print("filtering input")

        # The actual calculation
        filter = "data1"+operator+str(threshold)
        data1[eval(filter)] = pixel_value
        data1[data1 != pixel_value] = nodata

        if not silence:
            print("output result")

        # Write the out file
        driver = gdal.GetDriverByName(frmt)
        dsOut = driver.Create(outfile, ds1.RasterXSize, ds1.RasterYSize, 1, band1.DataType)
        gdar.CopyDatasetInfo(ds1, dsOut)
        bandOut = dsOut.GetRasterBand(1)
        bandOut.SetNoDataValue(nodata)
        gdar.BandWriteArray(bandOut, data1)

    except RuntimeError as err:
        raise err
    except Exception as e:
        raise e

    finally:
        # Close the datasets
        if band1: band1 = None
        if ds1: ds1 = None
        if bandOut: bandOut = None
        if dsOut: dsOut = None


def sieve_raster(infile, threshold, connectedness=4, outfile=None, mask=None, frmt="GTiff", silence=True):
    """    
    Removes raster polygons smaller than a provided threshold size (in pixels) and replaces 
    them with the pixel value of the largest neighbour polygon. The result can be written back 
    to the existing raster band, or copied into a new file.
    
    ###NOTE: Polygons smaller than the threshold with no neighbours that are as large as the threshold will not be altered. 
    Polygons surrounded by nodata areas will therefore not be altered.####

    The input dataset is read as integer data which means that floating point values are rounded 
    to integers. Re-scaling source data may be necessary in some cases 
    (e.g. 32-bit floating point data with min=0 and max=1).
      
    The algorithm makes three passes over the input file to enumerate the polygons and collect limited information about them. Memory use is proportional to the number of polygons (roughly 24 bytes per polygon), but is not directly related to the size of the raster. So very large raster files can be processed effectively if there aren't too many polygons. 
    But extremely noisy rasters with many one pixel polygons will end up being expensive (in memory) to process.

    More info:
    http://www.gdal.org/gdal__alg_8h.html#a33309c0a316b223bd33ae5753cc7f616
    http://www.gdal.org/gdal_sieve.html

    :param infile: path to input file
    :param threshold:  size threshold in pixels. Only raster polygons smaller than this size will be removed.
    :param connectedness: 
     4 Four connectedness should be used when determining polygons. That is diagonal pixels are not 
     considered directly connected. This is the default.

    8  Eight connectedness should be used when determining polygons. That is diagonal pixels are 
    considered directly connected
    
    :param outfile: path to output file, pass None to change the input in place
    :param mask: an optional path to a mask band. All pixels in the mask band with a value other than zero 
    will be considered suitable for inclusion in polygons.
    Pass "default" to use a mask included in the input file
    :param frmt: the output format, default "GTiff"
    :param silence: pass any value to enable debug info
    :return: None
    """

    if connectedness != 4:
        connectedness = 8

    src_ds = None
    dst_ds = None
    mask_ds = None

    try:
        #open source file and mask
        if not outfile:
            src_ds = gdal.Open(infile, gdal.GA_Update)
        else:
            src_ds = gdal.Open(infile, gdal.GA_ReadOnly)
        srcband = src_ds.GetRasterBand(1)

        if mask:
            if mask is 'default':
                maskband = srcband.GetMaskBand()
            else:
                mask_ds = gdal.Open(mask)
                maskband = mask_ds.GetRasterBand(1)
        else:
            maskband = None

        # define the output, copy properties from the input
        if outfile:
            drv = gdal.GetDriverByName(frmt)
            dst_ds = drv.Create(outfile, src_ds.RasterXSize, src_ds.RasterYSize, 1,srcband.DataType)
            wkt = src_ds.GetProjection()
            if wkt != '':
                dst_ds.SetProjection(wkt)
            dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
            dstband = dst_ds.GetRasterBand(1)
            # set the nodata value
            dstband.SetNoDataValue(srcband.GetNoDataValue())
        else:
            dstband = srcband

        if silence:
            prog_func = None
        else:
            prog_func = gdal.TermProgress

        gdal.SieveFilter(srcband, maskband, dstband,threshold, connectedness,callback=prog_func)

    except:
        if src_ds: src_ds = None
        if dst_ds: dst_ds = None
        if mask_ds: mask_ds = None


def get_wgs84_utm(lat,lon):
    """
    Return the WGS84 UTM EPSG code for a given wgs84 latitude/longitude
    see book "Postgis in Action Second Edition , p.74"
    :param lat:
    :param lon:
    :return:
    """
    if lat > 0:
        return 32600 + (math.floor((lon + 180) / 6) + 1)
    else:
        return 32700 + (math.floor((lon + 180) / 6) + 1)


if __name__ == "__main__":


    ### tests

    import os
    #os.chdir(r"K:\ITC\GFM2\2015\m13\code\dataXknnKmeans")
    #test projections
    etrs1989 = 3035
    rdnew = 28992
    wgs84 = 4326

    #gdal.UseExceptions()  # allow gdal exceptions

    def test_reproject_raster():

        ndviname = "2003.tif"

        inp = None
        out = None
        try:
            # create dataset
            inp = gdal.Open(ndviname)

            # reproject
            print("reprojecting to etrs1989...")
            out = reproject_raster(inp,epsg_from = None, epsg_to = etrs1989, fltr = gdal.GRA_Bilinear)
            # save to disk
            print("saving to disk...")
            inp.GetDriver().CreateCopy(ndviname+"_3035.tif", out)

            # reproject
            print("reprojecting to etrs1989...")
            out = reproject_raster_1(inp, pixel_spacing = None, epsg_from=None, epsg_to = etrs1989, fltr = gdal.GRA_Bilinear)
            # save to disk
            print("saving to disk...")
            inp.GetDriver().CreateCopy(ndviname+"_3035_1.tif", out)

            # reproject
            print("reprojecting to 4326...")
            out = reproject_raster(inp,epsg_from = None, epsg_to = 4326, fltr = gdal.GRA_Bilinear)
            # save to disk
            print("saving to disk...")
            inp.GetDriver().CreateCopy(ndviname+"_4326.tif", out)

            # reproject back
            print("reprojecting from 4326 to rdnew...")

            inp = gdal.Open(ndviname+"_4326.tif")
            out = reproject_raster(inp,epsg_from = None, epsg_to=28992, fltr = gdal.GRA_NearestNeighbour)
            # save to disk
            print("saving to disk...")
            inp.GetDriver().CreateCopy(ndviname+"_28992.tif", out)


        except RuntimeError as err:
            raise err
        except Exception as e:
             raise e

        finally:
            # close datasets
            if inp:
                inp = None
            if out:
                out = None

    #test_reproject_raster()


    def test_exportcsv():
        id = "afd3fcd91565847638e73034217121ff7"
        path = "/vagrant/code/pysdss/data/output/text/" + id + "/"
        inp = path+id+ "_keep.csv"
        if not os.path.exists(path+"shape"):
            os.mkdir(path+"shape")
        csv_to_polyline_shapefile(inp,"lat","lon","row", 32611, path+"shape"+"/row.shp", fastMethod=True)

    #test_exportcsv()

    def test_bufferlines():
        id = "afd3fcd91565847638e73034217121ff7"
        path = "/vagrant/code/pysdss/data/output/text/" + id + "/shape/"
        inp = path+"row.shp"
        buffer(inp,path+os.path.basename(inp).split(".")[0]+"_buffer1.shp",0.5)

    #test_bufferlines()


    def test_point_to_numbered_line():

        folder = "/vagrant/code/pysdss/data/output/text/afd3fcd91565847638e73034217121ff7/"
        file = folder +  "/shape/points.csv"
        arr = point_to_numbered_line(file, "Y", threshold=2, rowDirection="x")

        #file = folder + "/afd3fcd91565847638e73034217121ff7_keep.csv"
        #arr = point_to_numbered_line(file, "lat", threshold=2, rowDirection="x")

        df = pd.read_csv(file)

        x= np.hstack([df.values, arr.reshape(arr.size, 1)])
        print(x)
        np.savetxt("/vagrant/code/pysdss/data/output/text/afd3fcd91565847638e73034217121ff7/shape/points_update.csv",x,delimiter=",",fmt="%.1f")
    #test_point_to_numbered_line()


    def test_reclassify_raster():
        path = "/vagrant/geos_interp_experiments/data/"
        input = "keepc_invdist.tiff"
        threshold = 0.25
        operator = ">"

        if operator == ">":
            s = "more"
        elif operator == ">=":
            s = "more_equal"
        elif operator == "<":
            s = "less"
        elif operator == "<=":
            s = "less_equal"

        output = "keepc_invdist_" + s + str(threshold) + ".tiff"
        inputfile = path + input
        outputfile = path + output

        reclassify_raster(inputfile, outputfile, operator, threshold, silence=False)
   # test_reclassify_raster()


    def test_sieve():
        path = "/vagrant/geos_interp_experiments/data/"
        input = "keepc_invdist_more0.25.tiff"
        output = "keepc_invdist_more0.25_sieved.tiff"


        sieve_raster(path+input,50,8,path+output)#save to new file
        #sieve_raster(path + input, 50, 8, silence=False)#overwrite
    #test_sieve()


    def test_wgscode():
        print(get_wgs84_utm(38.42791286, -122.4100972))
        print(get_wgs84_utm(35.800211, -119.123065))

    #test_wgscode()
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        tiledClassify.py
# Purpose:      Tiled classification of an image given a sklearn classifier
#
# Author:      claudio piccinini
#
# Created:     12/09/2015
#-------------------------------------------------------------------------------

from osgeo import gdal
import numpy as np
import gdalRasterIO

def tiledClassification(img,classifier,  tilesize =(256,256), outname = 'classify.tif'):
    """
    Tiled classification of an image given a sklearn classifier
    :param img: image path
    :param classifier: scikit learn trained classifier
    :param tilesize: tile size (columns, rows)
    :param outname: output path
    :param use_indexed: use an indexed array (this is used to filter out pixels with nodata value)
    :return: None
    """

    out_ds = None
    out_band = None
    in_memory = None
    try:

        # we open the big image and get info
        # we get info from band 1; we assume all the bands have same info
        in_ds = gdal.Open(img)
        in_band = in_ds.GetRasterBand(1)
        xsize = in_band.XSize
        ysize = in_band.YSize
        nodata = in_band.GetNoDataValue()
        # how many bands is the input?
        nbands = in_ds.RasterCount


        # set properties of the classified image
        out_ds = in_ds.GetDriver().Create(outname, xsize, ysize, 1, in_band.DataType)
        out_ds.SetProjection(in_ds.GetProjection())
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())

        # we are creating a single band image so we access only band 1
        out_band = out_ds.GetRasterBand(1)

        # we set the tile size
        block_xsize, block_ysize = tilesize

        # we iterate the tiles by column and by row

        i = 0
        for x in range(0, xsize, block_xsize):
            if x + block_xsize < xsize:
                cols = block_xsize
            else:
                cols = xsize - x

            j = 0
            for y in range(0, ysize, block_ysize):
                if y + block_ysize < ysize:
                    rows = block_ysize
                else:  # C
                    rows = ysize - y
                print("reading tile row,col: %d %d / size row,col: %d %d / upperleft coord x,y: %d %d" % (j, i, rows,
                      cols, x, y))

                j += 1

                # read the multiband tile into a 3d numpy array
                data = in_ds.ReadAsArray(x, y, cols, rows)


                # Create memory target raster
                in_memory = gdal.GetDriverByName('MEM').Create('', cols, rows, nbands, in_band.DataType)
                in_memory.SetGeoTransform(in_ds.GetGeoTransform()) #assign a geotransform
                # Create for target raster the same projection as for the value raster
                in_memory.SetProjection(in_ds.GetProjection())

                # write data to the memory raster
                band = None
                for z in range(1, nbands+1):
                    band = in_memory.GetRasterBand(z)
                    band.WriteArray(data[z-1,:,:])
                    band.FlushCache()
                band = None

                # transform the inmemory multiband raster to an indexed array
                if nodata:
                    b = gdalRasterIO.raster_dataset_to_indexed_numpy(in_memory, "inmemory", nbands, "bycolumn", nodata)
                else: #if nodata is None just use the defaults
                    b = gdalRasterIO.raster_dataset_to_indexed_numpy(in_memory, "inmemory", nbands, "bycolumn")

                in_memory = None

                # use only the indexed array b[0] is the index, b[1] the indexed array,
                # b[2] the raster properties
                print("classify raster, wait...")
                y_pred = classifier.predict(b[1])

                # export back to raster
                # replace the indexed array with the classifiction result
                b=b[:1]+(y_pred,)+b[2:]
                # save the indexed array as an array, get a list of bands arrays,
                # rasterdt=gdalRasterIO.indexedNumpyToRasterDataset(b, outname+'.'+suffix, workingcatalog4 ,nodata=outnodatavalue, returnlist=True)

                if nodata:
                    rasterdt = gdalRasterIO.indexed_numpy_to_list(b, nodata=nodata)
                else:
                    rasterdt = gdalRasterIO.indexed_numpy_to_list(b)

                gg = rasterdt[0]

                # add the classified tile to the output
                out_band.WriteArray(gg, x, y)
                out_band.FlushCache()

            i += 1

        out_band.FlushCache()
        if nodata:
            out_band.SetNoDataValue(nodata)
        out_band.ComputeStatistics(False)
        # #out_ds.BuildOverviews('average', [2, 4, 8, 16, 32])
        out_ds = None

    finally:  # give control  back to C++
        if out_band:
            out_band = None
        if out_ds:
            out_ds = None
        if in_memory:
            in_memory = None
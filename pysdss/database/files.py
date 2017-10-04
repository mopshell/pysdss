# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        files.py
# Purpose:     operation on files
#
# Author:      claudio piccinini
#
# Updated:     04/10/2017
#-------------------------------------------------------------------------------

import os
import shapefile
import csv
import urllib.parse as parse

def get_fields(requestdata, UPLOAD_ROOT, UPLOAD_FORMATS ):

    """
    Get the fields for the uploaded file
    require folderid and fileformat csv or zip(shapefile) in the requestdara
    """

    '''
    Value | Shape Type
    0  | Null Shape
    1  | Point
    3  | PolyLine
    5  | Polygon
    8  | MultiPoint
    11 | PointZ
    13 | PolyLineZ
    15 | PolygonZ
    18 | MultiPointZ
    21 | PointM
    23 | PolyLineM
    25 | PolygonM
    28 | MultiPointM
    31 | MultiPatch
    '''

    try:
        idf = parse.unquote(requestdata.get('folderid'))
        ftype = parse.unquote(requestdata.get('fileformat'))

        if (idf and ftype):
            if (not os.path.exists(UPLOAD_ROOT + idf) or  ftype not in UPLOAD_FORMATS):
                raise Exception("request should contain a correct 'folderid' and/or 'fileformat'")

            if ftype == "csv":
                fields=["not_available"]
                #open csv and read first line
                files = os.listdir(UPLOAD_ROOT + idf)
                file = [i for i in files if i.endswith("csv")]

                if not file: raise Exception("a csv file was not found, check the request")

                f = open(UPLOAD_ROOT + idf + "/" +file[0])
                reader = csv.reader(f)
                fields += reader.__next__() #read just the first row
                f.close()

                return fields

            elif ftype == "zip":
                fields = ["not_available"]
                # open shape
                # read fields
                files = os.listdir(UPLOAD_ROOT + idf)
                file = [i.split(".")[0] for i in files if i.endswith("shp")]

                if not file: raise Exception("a shapefile was not found, check the request")

                #check this is a point
                sf = shapefile.Reader(UPLOAD_ROOT + idf +"/"+file[0])
                shapes = sf.shapes()
                if shapes[0].shapeType not in [1,11,21]:  #check point shapetypes, no multypoint
                    raise Exception("a point shapefile is required, upload a new file")

                return fields+[field[0].strip().lower() for field in sf.fields]

        else:
            raise Exception("request should contain a correct 'folderid' and/or 'fileformat'")

    except Exception as e:
        raise Exception(str(e))
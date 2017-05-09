# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        gridding.py
# Purpose:     code to prepare data for interpolation with gdal
#
# Author:      claudio piccinini
#
# Updated:     23/02/2017
#-------------------------------------------------------------------------------

import collections

def interpolate():
    raise NotImplementedError


def build_gdal_grid_string(params, outtext=True ,debug=False):
    """
    Build the strig for gdal_grid from a dictionary
    Output single string ot a list of strings
    http://www.gdal.org/gdal_grid.html

    :param params: a dictionary, leave value empty to skip the key, for "-q" you can pass anything


        params = {
            "-ot":"", "-of":"", "-co":"", "-zfield":"", "-z_increase":"",
            "-z_multiply":"lk", "-a_srs":"", "-spat":"", "-clipsrc":"pp",
            "-clipsrcsql":"", "-clipsrclayer":"", "-clipsrcwhere":"", "-l":"jj",
            "-where":"", "-sql":"", "-txe":"", "-tye":"","-outsize":"",
            "-a":"",
            "-q":"",
            "src_datasource":"",
            "dst_filename":""
        }

        To specify the algorithm pass a nested dictionary
        "-a":{
            name: ""
            param1:""
            param2:""
        }

        e.g.  "-a":{
                "name": "average",
                "radius1":"1",
                "radius2":"1",
                "angle":"0",
                "minpoints":"1",
                "nodata":"0"
        }
    :param outtext: True to return a string, otherwhise return a list
    :param debug: True to print the string on screen
    :return: a string or list with parameters
    """

    #TODO consider -l can be multiple layer names

    text=""
    for i in params:
        if params[i]:
            if i not in ["-q", "-a", "src_datasource","dst_filename"]:
                text+= i + " " + params[i] + " "

    #put some parameters at the end
    if params["-q"]: text+="-q "
    # build the algorithm string
    if params["-a"]:

        x=[]
        if params["-a"]["name"]: x.append(params["-a"]["name"])

        for i in params["-a"]:
            if i != "name": x.append( i+"="+params["-a"][i])
        text+="-a "+ ":".join(x) + " "

    if params["src_datasource"]: text += params["src_datasource"] + " "
    if params["dst_filename"]: text += params["dst_filename"]

    if debug:
        print(text)

    if outtext:
        return text
    else:
        return text.split(" ")


def build_gdal_rasterize_string(params, outtext=True ,debug=False):
    """
    Build the strig for gdal_rasterize from a dictionary
    Output single string ot a list of strings
    http://www.gdal.org/gdal_rasterize.html

    :param params: a dictionary, leave value empty to skip the key, for "-q" you can pass anything


        params = {
            "-b": "",
            "-i":"",
            "-at",
            "-burn":"",
            "-a": "",
            "-3d":"",
            "-add": "",
            "-l": "",
            "-where": "",
            "-sql":"",
            "-dialect":"",
            "-of":"",
            "-a_srs":"",
            "-co":"",
            "-a_nodata":"",
            "-init":"",
            "-a_srs":"",
            "-te":"",
            "-tr":"",
            "-tap":"",
            "-ts":"",
            "-ot":"",
            "-q":"",
            "src_datasource":"",
            "dst_filename":""
        }


    :param outtext: True to return a string, otherwhise return a list
    :param debug: True to print the string on screen
    :return: a string or list with parameters
    """

    text=""
    for i in params:
        if params[i]:
            if i not in ["-q", "src_datasource","dst_filename"]:
                text+= i + " " + params[i] + " "

    #put some parameters at the end
    if params["-q"]:
        text+="-q "

    if params["src_datasource"]:
        text += params["src_datasource"] + " "
    if params["dst_filename"]:
        text += params["dst_filename"]

    if debug:
        print(text)

    if outtext:
        return text
    else:
        return text.split(" ")


if __name__ == "__main__":

    ###Test#####


    ###test gdal_grid
    params = {
        "-ot":"",
        "-of":"",
        "-co":"",
        "-zfield":"",
        "-z_increase":"",
        "-z_multiply":"lk",
        "-a_srs":"",
        "-spat":"",
        "-clipsrc":"pp",
        "-clipsrcsql":"",
        "-clipsrclayer":"",
        "-clipsrcwhere":"",
        "-l":"jj",
        "-where":"",
        "-sql":"",
        "-txe":"",
        "-tye":"",
        "-outsize":"",
        "-a":{
                "name": "average",
                "radius1":"1",
                "radius2":"1",
                "angle":"0",
                "minpoints":"1",
                "nodata":"0"
        },
        "-q":"",
        "src_datasource":"",
        "dst_filename":""
    }

    a = build_gdal_grid_string(params, False)

    print(a)

    ###test gdal_rasterize


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
        "-te": "xmin ymin xmax ymax",
        "-tr": "xres yres",
        "-tap": "",
        "-ts": "width height",
        "-ot": "Byte",
        "-q": "",
        "src_datasource": "",
        "dst_filename": ""
    }

    a = build_gdal_rasterize_string(params,outtext=True)

    print(a)




# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:        database.py
# Purpose:     synchronous/asynchronous functions called via celery by the django rest interface
#
# Author:      claudio piccinini
#
# Updated:     04/10/2017
# -------------------------------------------------------------------------------

import time
from pysdss.database import query
from pysdss.database import files

def upload_metadata(*args, **kw):
    """
    Receive arguments from the django request and call method to upload metadata to database
    :param args:
    :param kw:
    :return:
        return dictionary, content follows the rest api specification
        {"success": True, "content": [{"name": "", "type": "", "value": ""}, ...]}
        {"success": False, "content": "<errorsstring>"}

    ## tests
    # 1 parameters missing
    # curl -X POST -H 'Content-Type:multipart/form-data' -F 'file=@//vagrant/code/pysdss/data/input/2016_06_17.csv' http://localhost:8000/processing/database/upload/executesync/
    # 2wrong format
    # curl -X POST -H 'Content-Type:multipart/form-data' -F 'file=@//vagrant/code/pysdss/data/input/2016_06_17.txt' -F 'metatable=sensor' -F 'toolid=1' -F 'datetime=2017-08-15%2014:19:25.63' -F 'roworientation=NE' http://localhost:8000/processing/database/upload/executesync/
    # 3OK
    # curl -X POST -H 'Content-Type:multipart/form-data' -F 'file=@//vagrant/code/pysdss/data/input/shape.zip'  -F 'metatable=sensor' -F 'toolid=1' -F 'datetime=2017-08-15%2014:19:25.63' -F 'roworientation=NE' http://localhost:8000/processing/database/upload/executesync/
    # 4 wrong zip content
    # curl -X POST -H 'Content-Type:multipart/form-data' -F 'file=@//vagrant/code/pysdss/data/input/badshape.zip' -F 'metatable=sensor' -F 'toolid=1' -F 'datetime=2017-08-15%2014:19:25.63' -F 'roworientation=NE' http://localhost:8000/processing/database/upload/executesync/
    # 5 OK
    # curl -X POST -H 'Content-Type:multipart/form-data' -F 'file=@//vagrant/code/pysdss/data/input/2016_06_17.csv' -F 'metatable=sensor' -F 'toolid=1' -F 'datetime=2017-08-15%2014:19:25.63' -F 'roworientation=NE' http://localhost:8000/processing/database/upload/executesync/

    """


    try:

        # iddataset = query.upload_metadata(request.data, settings.METADATA_ID_TYPES, settings.METADATA_FIELDS_TYPES, settings.METADATA_IDS)

        # get additional keys and delete them before calling the method, did thid to respect the legacy method signature
        idf = kw['idf']
        fl_ext = kw['fl_ext']
        METADATA_ID_TYPES = kw['METADATA_ID_TYPES']
        METADATA_FIELDS_TYPES = kw['METADATA_FIELDS_TYPES']
        METADATA_IDS = kw['METADATA_IDS']
        del kw['METADATA_ID_TYPES']
        del kw['METADATA_FIELDS_TYPES']
        del kw['METADATA_IDS']
        del kw['idf']
        del kw['fl_ext']

        iddataset = query.upload_metadata(kw, METADATA_ID_TYPES, METADATA_FIELDS_TYPES, METADATA_IDS)

        # return Response(up_file.name, status.HTTP_201_CREATED)
        # return the folderID, the metadata newrow ID, and the file extension
        # return Response({"success": True, "content": [idf, iddataset, up_file.name.split('.')[-1]]})
        # return {"success": True, "content": [idf, iddataset, fl_ext]}

        return {"success": True, "content": [
            {"name": "folderid", "type": "string", "value": idf},
            {"name": "datasetid", "type": "number", "value": iddataset},
            {"name": "fileformat", "type": "string", "value": fl_ext}]}

    except Exception as e:
        # return Response({"success": False, "content": str(e)})  # errors coming from query.upload_metadata
        return {"success": False, "content": str(e)}  # errors coming from query.upload_metadata


def get_fields(*args, **kw):
    """
    Receive arguments from the django request and call method to upload metadata to database
    :param args:
    :param kw:
    :return:
        return dictionary, content follows the rest api specification
        {"success": True, "content": [{"name": "", "type": "", "value": ""}, ...]}
        {"success": False, "content": "<errorsstring>"}

    curl -X  POST -H  'Content-Type:multipart/form-data' -F 'fileformat=zip' -F 'folderid=a5f9e0915ecb94449b26a8dc52b970cc0'   http://localhost:8000/processing/database/getfields/executesync/


    """

    try:

        # get additional keys and delete them before calling the method, did this to respect the legacy method signature
        UPLOAD_ROOT = kw['UPLOAD_ROOT']
        UPLOAD_FORMATS = kw['UPLOAD_FORMATS']
        del kw['UPLOAD_ROOT']
        del kw['UPLOAD_FORMATS']

        fieldnames = files.get_fields(kw,UPLOAD_ROOT, UPLOAD_FORMATS)

        return {"success": True, "content": [
            {"name": "fieldnames", "type": "list", "value": fieldnames}]}

    except Exception as e:
        # return Response({"success": False, "content": str(e)})  # errors coming from query.upload_metadata
        return {"success": False, "content": str(e)}  # errors coming from query.upload_metadata


def upload_data(*args, **kw):
    """
    Receive arguments from the django request and call method to upload metadata to database
    :param args:
    :param kw:
    :return:
        return dictionary, content follows the rest api specification
        {"success": True, "content": [{"name": "", "type": "", "value": ""}, ...]}
        {"success": False, "content": "<errorsstring>"}





    curl -X  POST -H  'Content-Type:multipart/form-data' -F 'metatable=canopy' -F 'row=sensor_add' -F 'filename=ToKalonNDVI.zip' -F 'folderid=a5f9e0915ecb94449b26a8dc52b970cc0' -F 'datasetid=2' -F 'lat=lat' -F 'lon=lng' -F 'value1=sf01' -F 'value2=sf02' -F 'value3=sf03' -F 'value4=sf04' -F 'value5=course' -F 'value6=speed' http://localhost:8000/processing/database/todatabase/executesync/

    curl -X  POST -H  'Content-Type:multipart/form-data' -F 'metatable=canopy' -F 'row=sensor_addr' -F 'filename=2017-07-25 To Kalon NDVI.csv' -F 'folderid=a5f9e0915ecb94449b26a8dc52b970cc0' -F 'datasetid=2' -F 'lat=lat' -F 'lon=lng' -F 'value1=sf01' -F 'value2=sf02' -F 'value3=sf03' -F 'value4=sf04' -F 'value5=course' -F 'value6=speed' http://localhost:8000/processing/database/todatabase/executesync/

    """

    try:

        # get additional keys and delete them before calling the method, did thid to respect the legacy method signature
        METADATA_DATA_TABLES = kw['METADATA_DATA_TABLES']
        METADATA_IDS = kw['METADATA_IDS']
        UPLOAD_ROOT = kw['UPLOAD_ROOT']
        del kw['METADATA_DATA_TABLES']
        del kw['METADATA_IDS']
        del kw['UPLOAD_ROOT']

        query.upload_data(kw, METADATA_DATA_TABLES, METADATA_IDS,UPLOAD_ROOT)

        # return Response(up_file.name, status.HTTP_201_CREATED)
        # return the folderID, the metadata newrow ID, and the file extension
        # return Response({"success": True, "content": [idf, iddataset, up_file.name.split('.')[-1]]})
        # return {"success": True, "content": [idf, iddataset, fl_ext]}

        return {"success": True, "content": [{"name": "response", "type": "string", "value": "data was uploaded"}]}

    except Exception as e:
        # return Response({"success": False, "content": str(e)})  # errors coming from query.upload_metadata
        return {"success": False, "content": str(e)}  # errors coming from query.upload_metadata


def get_json(*args, **kw):
    """
    Receive arguments from the django request and call method to upload metadata to database
    :param args:
    :param kw:
    :return:
        return dictionary, content follows the rest api specification
        {"success": True, "content": [{"name": "", "type": "", "value": ""}, ...]}
        {"success": False, "content": "<errorsstring>"}

    curl -X  POST -H  'Content-Type:multipart/form-data' -F 'metatable=canopy'  -F 'datasetid=2' http://localhost:8000/processing/database/getjson/executesync/
    curl -X  POST -H  'Content-Type:multipart/form-data' -F 'metatable=canopy'  -F 'datasetid=2' -F 'values=value1,value6' http://localhost:8000/processing/database/getjson/executesync/

    """

    try:

        # get additional keys and delete them before calling the method, did this to respect the legacy method signature


        METADATA_DATA_TABLES= kw['METADATA_DATA_TABLES']
        METADATA_IDS= kw['METADATA_IDS']
        DATA_IDS= kw['DATA_IDS']

        del kw['METADATA_DATA_TABLES']
        del kw['METADATA_IDS']
        del kw['DATA_IDS']

        gjson = query.get_geojson(kw,METADATA_DATA_TABLES,METADATA_IDS,DATA_IDS  )

        return {"success": True, "content": [
            {"name": "geojson", "type": "json", "value": gjson}]}

    except Exception as e:
        # return Response({"success": False, "content": str(e)})  # errors coming from query.upload_metadata
        return {"success": False, "content": str(e)}  # errors coming from query.upload_metadata


def get_vmap(*args, **kw):
    """
    Receive arguments from the django request and call method to upload metadata to database
    :param args:
    :param kw:
    :return:
        return dictionary, content follows the rest api specification
        {"success": True, "content": [{"name": "", "type": "", "value": ""}, ...]}
        {"success": False, "content": "<errorsstring>"}

    curl -X  POST -H  'Content-Type:multipart/form-data' -F 'metatable=canopy'  -F 'datasetid=2' http://localhost:8000/processing/database/getvmap/executesync/

    """

    try:

        # get additional keys and delete them before calling the method, did this to respect the legacy method signature
        METADATA_IDS= kw['METADATA_IDS']
        del kw['METADATA_IDS']

        vmap = query.get_vmap(kw, METADATA_IDS)
        return {"success": True, "content": [
            {"name": "vmap", "type": "json", "value": vmap}]}

    except Exception as e:
        # return Response({"success": False, "content": str(e)})  # errors coming from query.upload_metadata
        return {"success": False, "content": str(e)}  # errors coming from query.upload_metadata


def upload_ids(*args, **kw):
    """
    Receive arguments from the django request and call method to upload metadata to database
    :param args:
    :param kw:
    :return:
        return dictionary, content follows the rest api specification
        {"success": True, "content": [{"name": "", "type": "", "value": ""}, ...]}
        {"success": False, "content": "<errorsstring>"}

    curl -X  POST -H  'Content-Type:multipart/form-data' -F 'metatable=canopy'  -F 'datasetid=2' -F 'ids=439689,439691,439692' -F 'folderid=a5f9e0915ecb94449b26a8dc52b970cc0' -F 'exclude=true' http://localhost:8000/processing/database/uploadids/executesync/
    curl -X  POST -H  'Content-Type:multipart/form-data' -F 'metatable=canopy'  -F 'datasetid=2' -F 'ids=439689,439691,439692' -F 'folderid=a5f9e0915ecb94449b26a8dc52b970cc0' http://localhost:8000/processing/database/uploadids/executesync/

    """

    try:

        # get additional keys and delete them before calling the method, did this to respect the legacy method signature
        METADATA_DATA_TABLES = kw['METADATA_DATA_TABLES']
        METADATA_IDS= kw['METADATA_IDS']
        DATA_IDS = kw['DATA_IDS']
        UPLOAD_ROOT = kw['UPLOAD_ROOT']
        del kw['METADATA_DATA_TABLES']
        del kw['METADATA_IDS']
        del kw['DATA_IDS']
        del kw['UPLOAD_ROOT']

        folderid, filename, epsg = query.select_data(kw,METADATA_DATA_TABLES, METADATA_IDS, DATA_IDS, UPLOAD_ROOT)

        return {"success": True, "content": [
            {"name": "folderid", "type": "string", "value": folderid},
            {"name": "filename", "type": "string", "value": filename},
            {"name": "epsg", "type": "number", "value": epsg}]}

    except Exception as e:
        # return Response({"success": False, "content": str(e)})  # errors coming from query.upload_metadata
        return {"success": False, "content": str(e)}  # errors coming from query.upload_metadata
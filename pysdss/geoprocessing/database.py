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

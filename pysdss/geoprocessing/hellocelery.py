# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        hellocelery.py
# Purpose:     testing synchronous/asynchronous processing via celery
#               these test functions are called by the django rest interface
#
# Author:      claudio piccinini
#
# Updated:     22/05/2017
#-------------------------------------------------------------------------------

import time
from pysdss.database import query

def synchronous_test_1(*args, **kw):

    """
  
    :return:   
    functions must return a list of dictionaries, each dictionary for each result in case of
    multiple outputs
    [{"name":"", "type":"string"|"number"|"url", "value":""}, {...}, ...]
    
    """
    #time.sleep(3)
    #raise Exception(["error","error info here"]) # ["error"] will be the value for result.get()
    #return [{"name":"", "type":"string", "value":"synch finished"}]
    #return [{"name":"", "type":"string", "value":[arg for arg in args]}]

    #print(kw)
    return [{"name":"", "type":"string", "message":"synch finished", "value":[kw]}]


def asynchronous_test_1(*args, **kw):
    #raise Exception(["error","error info here"]) # ["error"] will be the value for result.get()
    time.sleep(15)
    return [{"name": "", "type": "string", "message": "asynch finished", "value":[kw]}]


def synchronous_test_upload(*args, **kw):
    """

    :return:
    functions must return a list of dictionaries, each dictionary for each result in case of
    multiple outputs
    [{"name":"", "type":"string"|"number"|"url", "value":""}, {...}, ...]

    """
    # time.sleep(3)
    # raise Exception(["error","error info here"]) # ["error"] will be the value for result.get()
    # return [{"name":"", "type":"string", "value":"synch finished"}]
    # return [{"name":"", "type":"string", "value":[arg for arg in args]}]

    # print(kw)
    #print(args)

    try:

        #iddataset = query.upload_metadata(request.data, settings.METADATA_ID_TYPES, settings.METADATA_FIELDS_TYPES, settings.METADATA_IDS)

        # get additional keys and delete them before calling the method, did thid to respect the legacy method signature
        idf = kw['idf']
        fl_ext = kw['fl_ext']
        METADATA_ID_TYPES = kw['METADATA_ID_TYPES']
        METADATA_FIELDS_TYPES  = kw['METADATA_FIELDS_TYPES']
        METADATA_IDS = kw['METADATA_IDS']
        del kw['METADATA_ID_TYPES']
        del kw['METADATA_FIELDS_TYPES']
        del kw['METADATA_IDS']
        del kw['idf']
        del kw['fl_ext']

        iddataset = query.upload_metadata(kw, METADATA_ID_TYPES, METADATA_FIELDS_TYPES, METADATA_IDS)

        # return Response(up_file.name, status.HTTP_201_CREATED)
        # return the folderID, the metadata newrow ID, and the file extension
        #return Response({"success": True, "content": [idf, iddataset, up_file.name.split('.')[-1]]})
        #return {"success": True, "content": [idf, iddataset, fl_ext]}

        return {"success": True, "content": [
            {"name":"folderid","type":"string","value":idf },
            {"name":"datasetid","type":"number","value":iddataset},
            {"name":"fileformat","type":"string","value":fl_ext}]}

    except Exception as e:
        #return Response({"success": False, "content": str(e)})  # errors coming from query.upload_metadata
        return {"success": False, "content": str(e)}  # errors coming from query.upload_metadata

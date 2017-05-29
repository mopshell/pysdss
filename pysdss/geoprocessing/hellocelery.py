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

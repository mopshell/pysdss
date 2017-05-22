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

def synchronous_test_1():
    time.sleep(3)
    return "synch finished"

def asynchronous_test_1():
    raise Exception("errore!") # "errore!" will be the value for result.get()
    time.sleep(30)
    return "asynch finished"

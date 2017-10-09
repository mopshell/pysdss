# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        query.py
# Purpose:     query postgresql database to select/add/delete data
#
# Author:      claudio piccinini
#
# Updated:     10/08/2017
#-------------------------------------------------------------------------------

import urllib.parse as parse
import csv
import math
import os.path

import psycopg2
from psycopg2.extensions import AsIs
import shapefile

from pysdss.settings import default_connection
from pysdss.database import pgconnect

import pysdss.utility.utils as utils


##############################

def get_records(table, fields, conndict=None, cur=None):
    """
    return the id and a field
    :param table: table to query , can be YieldType, SoilType, CanoType, SensoType
    :param fields: list od fields to query
    :param conndict: dictionary with connection parameters
    :param cur: psycopg2 cursor
    :return: a list of values

    """

    cur=cur
    conn=None
    hadcursor = True if cur else False #set if we are using an already existing connection

    try:
        if not cur:
            conn = pgconnect(**conndict)
            cur = conn.cursor()
        cur.execute("SELECT %s,%s FROM %s;", (AsIs(fields[0]), AsIs(fields[1]), AsIs(table))) #AsIs will hide the quotes
        #print(result)
        return cur.fetchall()


    except Exception as e:
        print(e)
        raise(e)

    finally:
        if not hadcursor: #if we passed the cursor we will not close the connection in this function
            if cur:
                cur.close()
            if conn:
                conn.close()

def set_record(table, field, value, conndict=None, cur=None):
    """
    insert a record with single field
    :param table: table to query , can be YieldType, SoilType, CanoType, SensoType
    :param tpe: string with the type to add
    :param conndict: dictionary with connection parameters
    :param cur: psycopg2 cursor
    :return: a list of values

    """

    cur = cur
    conn = None
    hadcursor = True if cur else False  # set if we are using an already existing connection
    result = []
    try:
        if not cur:
            conn = pgconnect(**conndict)
            cur = conn.cursor()
        cur.execute("INSERT INTO %s (%s) VALUES (%s);", (AsIs(table),AsIs(field),value))  # AsIs will hide the quotes
        conn.commit()

    except Exception as e:
        print(e)
        raise(e)

    finally:
        if not hadcursor:  # if we passed the cursor we will not close the connection in this function
            if cur:
                cur.close()
            if conn:
                conn.close()

def delete_record(table, field, value, conndict=None, cur=None):
    """
    Delete a record given a field value
    :param table:
    :param field: field to query
    :param value: query value
    :param conndict: pycopg2 with dictionary connection parameters
    :param cur: cursor
    :return:
    """

    cur = cur
    conn = None
    hadcursor = True if cur else False  # set if we are using an already existing connection
    result = []
    try:
        if not cur:
            conn = pgconnect(**conndict)
            cur = conn.cursor()
        cur.execute("DELETE FROM  %s  WHERE %s=%s;", (AsIs(table), AsIs(field), value))  # AsIs will hide the quotes
        conn.commit()

    except Exception as e:
        print(e)
        raise (e)

    finally:
        if not hadcursor:  # if we passed the cursor we will not close the connection in this function
            if cur:
                cur.close()
            if conn:
                conn.close()

###################################


def upload_metadata(requestdata, METADATA_ID_TYPES,METADATA_FIELDS_TYPES,METADATA_IDS, conndict=None, cur=None):
    """
    insert row in the metadata table and return the row ID
    :param requestdata: the dictionary with data
    :param METADATA_ID_TYPES: mapping between metadata table and idtype field name
    :param METADATA_FIELDS_TYPES: mapping between field name and its type('string' or 'number'), this will be used to
        `                           build the correct insert query string
    :param METADATA_IDS: mapping between the metadata table name and the id name
    :param conndict: dictionary with connection properties, if None the default connection will be used
    :param cur: psycopg2 cursor
    :return:  the ID of the inserted row
    """



    metatable = parse.unquote(requestdata.get("metatable"))
    toolid = parse.unquote((requestdata.get("toolid")))
    toolidname = METADATA_ID_TYPES[metatable] #get the field name for the tool

    cur = cur
    conn = None
    hadcursor = True if cur else False  # set if we are using an already existing connection



    try:
        if not cur:
            if not conndict:conndict=default_connection
            conn = pgconnect(**conndict)
            cur = conn.cursor()

        #get fields and values to build the insert query
        fields = toolidname
        values = toolid


        if len(requestdata)>2: # 2 fields are metatable and toolid
            for i in requestdata:
                if (i not in ["metatable","toolid","file","valuemap"]):  #file is the uploaded file coming from the POST request
                    fields += "," + i
                    if METADATA_FIELDS_TYPES[i]=="string":
                        values += ",'" + parse.unquote(requestdata[i])+"'"
                    else:
                        values += "," + parse.unquote(requestdata[i])
        fields += ",valuemap"
        values += ",'{}'"

        #cur.execute("INSERT INTO %s (%s) VALUES (%s);", (AsIs(metatable),AsIs(field),value))  # AsIs will hide the quotes
        #build the query string, this may be a security problem ....
        cur.execute("INSERT INTO " + metatable + " (" + fields + ") VALUES (" + values + ") RETURNING " + METADATA_IDS[metatable] + ";",())
        iddataset = cur.fetchone()[0]
        conn.commit()

        return iddataset

    except Exception as e:
        raise Exception("error adding metadata into the database, try to upload the file again" + str(e))


    finally:
        if not hadcursor:  # if we passed the cursor we will not close the connection in this function
            if cur:
                cur.close()
            if conn:
                conn.close()

#todo check the upload for sensodatum
def upload_data(requestdata,METADATA_DATA_TABLES, METADATA_IDS,UPLOAD_ROOT, proj=4326, conndict=None, cur=None):
    """
    Upload data to tables yielddatum, soildatum, canodatum, sensodatum
    :param requestdata:  dictionary with request parameters

            for example  {"metatable":"canopy", "filename":"2017-07-25 To Kalon NDVI.csv", "folderid":"a5f9e0915ecb94449b26a8dc52b970cc0",
            "datasetid": "1", "lat": "lat", "lon": "lng", "value1":"sf01","row": "sensor_addr"}

    :param METADATA_DATA_TABLES: mapping between metadata tables names and  dataset table names
    :param METADATA_IDS: id fields names for metadata tables
    :param conndict: dictionary with connection properties, if None the default connection will be used
    :param cur: psycopg2 cursor
    :return: True if everything is ok
    """

    tablename = METADATA_DATA_TABLES[parse.unquote(requestdata.get('metatable'))]   # e.g. canopy->canodatum
    metaid = METADATA_IDS[parse.unquote(requestdata.get('metatable'))]   # e.g. canopy->id_canopy


    #checking if there are value2 to value6


    cur = cur
    conn = None
    hadcursor = True if cur else False  # set if we are using an already existing connection


    try:
        if not cur:
            if not conndict:
                conndict=default_connection
            conn = pgconnect(**conndict)
            cur = conn.cursor()

        conn.autocommit = False

        #todo

        #call functions to update the valuemap and insert data from the csv or shapefile
        if parse.unquote(requestdata.get('filename')).split('.')[-1] == "csv":
            upload_data_csv(cur,requestdata,tablename, metaid, UPLOAD_ROOT+"/"+parse.unquote(requestdata.get("folderid"))+"/"+parse.unquote(requestdata.get("filename")))
        elif parse.unquote(requestdata.get('filename')).split('.')[-1] == "zip":
            upload_data_shape(cur,requestdata,tablename, metaid, UPLOAD_ROOT+"/"+parse.unquote(requestdata.get("folderid"))+"/"+parse.unquote(requestdata.get("filename")))


    except Exception as e:
        raise Exception("error adding data into the database, try to upload the file again: " + str(e))


    else:
        #insert geometry and vacuumanalyze
        #print("total rows: " + str(i))
        conn.commit() #commits the previous update/inserts

        try:
            conn.autocommit = True  # necessary for vacuum
            print("creating geometry")
            ## fill geometry column
            cur.execute(
                """UPDATE data."""+tablename+""" SET geom = ST_PointFromText('POINT(' || lon || ' ' || lat || ')', %s)""",
                (proj,))
            print("creating index")
            ## create geo index
            try:  # create spatial index if it does not exists
                cur.execute("""CREATE INDEX idx_"""+tablename+"""_geom ON data."""+tablename+""" USING GIST(geom)""")
            except Exception as e:
                print(e)  # this will happen if the spatial index is already there

            print("vacuum analize clustering")
            cur.execute("""VACUUM ANALYZE data."""+tablename)
            #cur.execute("""VACUUM ANALYZE data.Sensor""")
            #cur.execute("""VACUUM ANALYZE data.ColorGrade""")
            #cur.execute("""VACUUM ANALYZE data.BerryCount""")
            #cur.execute("""VACUUM ANALYZE data.BerrySize""")
            cur.execute("""CLUSTER data."""+tablename+""" USING idx_"""+tablename+"""_geom""")
            cur.execute("""ANALYZE data."""+tablename)

        except Exception as e:
            #todo here we should delete the data already inserted? i think no, we should create a function that run somethimes and update the geometry field
            raise Exception("error adding data into the database, try to upload the file again: " + str(e))

    finally:
        if not hadcursor:  # if we passed the cursor we will not close the connection in this function
            if cur:
                cur.close()
            if conn:
                conn.close()


def prepare_upload_statements(requestdata, cursor, tablename, metaid):
    """
    Prepare the statements to speedup data insertion
    :param requestdata:dictionary with request data
    :param cursor: psycopg2 cursor
    :param tablename: name for the metadata database table
    :param metaid: name of the id column for the metadata table
    :return:
    """
    if requestdata.get('value1') == "not_available":
        raise Exception("value1 cannot be set to 'not_available'")

    types = ""
    clnames = ""
    values = ""

    if requestdata.get("row"): #this is for inserting the $n
        start = 6
    else:
        start = 5

    for i in "23456":
        if requestdata.get("value"+i):
            types += ",double precision"
            clnames += ",value"+i
            values += ",$"+str(start)
            start += 1

    # prepare the statement to speed up the insertion
    if requestdata.get("row"):  # noparse ,not needed here as  it would raise error if None
        cursor.execute("""PREPARE """ + tablename + """(integer,smallint, double precision, double precision, 
        double precision"""+types+""") AS INSERT INTO data.""" + tablename + """(""" + metaid + """,row, lat,lon, value1"""+clnames+""")
        VALUES ($1,$2,$3, $4, $5"""+values+""")""")

    else:
        cursor.execute("""PREPARE """ + tablename + """(integer, double precision, double precision, 
        double precision"""+types+""") AS INSERT INTO data.""" + tablename + """(""" + metaid + """, lat,lon, value1"""+clnames+""")
        VALUES ($1,$2,$3, $4"""+values+""")""")


def update_valuemap(cur, requestdata,metaid, nvalues=0):
    """
    Update the valuemap in metadata tables Sensor, Soil, Yield, Canopy
    :param cur:psycopg2 cursor
    :param requestdata: dictionary with request data
    :param metaid: name of the id column for the metadata table
    :param nvalues: available value fields beyond value1
    :return:
    """

    table = parse.unquote(requestdata.get('metatable'))
    datasetid = parse.unquote(requestdata['datasetid'])

    #build json valuemap to map value columns to the file columns and add to the database
    jsn = "{"
    value1 = parse.unquote(requestdata['value1'])
    jsn += '"value1":"'+value1+'"'

    if nvalues > 0:
        for i in range(2, nvalues+2):
            jsn += ',"value'+str(i)+'":"' + parse.unquote(requestdata["value"+str(i)]) + '"'

    jsn += "}"

    #return """UPDATE """+table+""" SET valuemap='"""+jsn+"""' WHERE id_sensor="""+datasetid+""";"""

    # commit will be done by the caller
    cur.execute("""UPDATE """+table+""" SET valuemap='"""+jsn+"""' WHERE """+metaid+"""="""+datasetid+""";""")


def upload_data_csv(cur, requestdata, tablename, metaid, filepath, debug=False):
    """
    Open csv file and upload to postgresql
    :param cur: psycopg2 cursor
    :param requestdata: dictionary with request data
    :param tablename: name for the metadata database table
    :param metaid: name of the id column for the metadata table
    :param filepath: full path to csv file
    :param debug: True to print out info
    :return: None
    """

    print("uploading csv data")
    i = 0

    with open(filepath) as f:
        reader = csv.reader(f)
        #get the filed names, and strio/lower them, i did not use dictreader because of the csv table with strip problems
        columns = reader.__next__()
        columns = [a.strip().lower() for a in columns]

        #gey id for current dataset
        datasetid = parse.unquote(requestdata['datasetid'])
        #get column name index for current csv file, the parameters values are the field names
        idlat = columns.index(parse.unquote(requestdata['lat']))
        idlon = columns.index(parse.unquote(requestdata['lon']))
        idvalue1 = columns.index(parse.unquote(requestdata['value1']))


        #IMPORTANT   the code below works if the client sends the valuen columns in order!
        nvalues = 0
        while True:
            if requestdata.get("value2"):
                idvalue2 = columns.index(parse.unquote(requestdata['value2']));nvalues+=1
            else: break
            if requestdata.get("value3"):
                idvalue3 = columns.index(parse.unquote(requestdata['value3']));nvalues+=1
            else: break
            if requestdata.get("value4"):
                idvalue4 = columns.index(parse.unquote(requestdata['value4']));nvalues+=1
            else: break
            if requestdata.get("value5"):
                idvalue5 = columns.index(parse.unquote(requestdata['value5']));nvalues+=1
            else: break
            if requestdata.get("value6"):
                idvalue6 = columns.index(parse.unquote(requestdata['value6']));nvalues+=1
            break

        #prepare the statement to speed up the insertion
        print("updating field valuemap and prepare upload statements")
        update_valuemap(cur, requestdata,metaid, nvalues)
        prepare_upload_statements(requestdata, cur, tablename, metaid)

        if nvalues == 0:
            if requestdata.get("row"):
                idrow = columns.index(parse.unquote(requestdata['row']))

                for line in reader:
                    #print(datasetid, line[idrow], line[idlat],line[idlon], line[idvalue])
                    cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s)""",
                                (datasetid, line[idrow], line[idlat],line[idlon], line[idvalue1]))

                    #raise Exception("la cucaracia with row!")
                    if debug:
                        i += 1
                        if i % 5000 == 0:
                            print(i)
            else:
                for line in reader:
                    #print(datasetid, line[idlat], line[idlon], line[idvalue])
                    cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s)""",
                                (datasetid, line[idlat], line[idlon], line[idvalue1]))

                    #raise Exception("la cucaracia!")
                    if debug:
                        i += 1
                        if i % 5000 == 0:
                            print(i)

        elif nvalues == 1:
            if requestdata.get("row"):
                idrow = columns.index(parse.unquote(requestdata['row']))

                for line in reader:
                    # print(datasetid, line[idrow], line[idlat],line[idlon], line[idvalue])
                    cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s)""",
                                (datasetid, line[idrow], line[idlat], line[idlon], line[idvalue1],line[idvalue2]))
            else:
                for line in reader:
                    # print(datasetid, line[idlat], line[idlon], line[idvalue])
                    cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s,%s)""",
                                (datasetid, line[idlat], line[idlon], line[idvalue1],line[idvalue2]))
        elif nvalues == 2:
            if requestdata.get("row"):
                idrow = columns.index(parse.unquote(requestdata['row']))

                for line in reader:
                    # print(datasetid, line[idrow], line[idlat],line[idlon], line[idvalue])
                    cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s, %s)""",
                                (datasetid, line[idrow], line[idlat], line[idlon], line[idvalue1],line[idvalue2],line[idvalue3]))
            else:
                for line in reader:
                    # print(datasetid, line[idlat], line[idlon], line[idvalue])
                    cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s,%s, %s)""",
                                (datasetid, line[idlat], line[idlon], line[idvalue1],line[idvalue2],line[idvalue3]))
        elif nvalues == 3:
            if requestdata.get("row"):
                idrow = columns.index(parse.unquote(requestdata['row']))

                for line in reader:
                    # print(datasetid, line[idrow], line[idlat],line[idlon], line[idvalue])
                    cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s, %s, %s)""",
                                (datasetid, line[idrow], line[idlat], line[idlon], line[idvalue1],line[idvalue2],line[idvalue3],line[idvalue4]))
            else:
                for line in reader:
                    # print(datasetid, line[idlat], line[idlon], line[idvalue])
                    cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s,%s, %s, %s)""",
                                (datasetid, line[idlat], line[idlon], line[idvalue1],line[idvalue2],line[idvalue3],line[idvalue4]))
        elif nvalues == 4:
            if requestdata.get("row"):
                idrow = columns.index(parse.unquote(requestdata['row']))

                for line in reader:
                    # print(datasetid, line[idrow], line[idlat],line[idlon], line[idvalue])
                    cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s, %s, %s, %s)""",
                                (datasetid, line[idrow], line[idlat], line[idlon], line[idvalue1],line[idvalue2],line[idvalue3],line[idvalue4],line[idvalue5]))
            else:
                for line in reader:
                    # print(datasetid, line[idlat], line[idlon], line[idvalue])
                    cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s,%s, %s, %s, %s)""",
                                (datasetid, line[idlat], line[idlon], line[idvalue1],line[idvalue2],line[idvalue3],line[idvalue4],line[idvalue5]))
        elif nvalues == 5:
            if requestdata.get("row"):
                idrow = columns.index(parse.unquote(requestdata['row']))

                for line in reader:
                    # print(datasetid, line[idrow], line[idlat],line[idlon], line[idvalue])
                    cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s, %s, %s, %s, %s)""",
                                (datasetid, line[idrow], line[idlat], line[idlon], line[idvalue1],line[idvalue2],line[idvalue3],line[idvalue4],line[idvalue5],line[idvalue6]))
            else:
                for line in reader:
                    # print(datasetid, line[idlat], line[idlon], line[idvalue])
                    cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s,%s, %s, %s, %s, %s)""",
                                (datasetid, line[idlat], line[idlon], line[idvalue1],line[idvalue2],line[idvalue3],line[idvalue4],line[idvalue5],line[idvalue6]))


#todo check shp2pgsql
def upload_data_shape(cur, requestdata,tablename,  metaid, filepath, debug=False):
    """
    Open shapefile and upload to postgresql
    :param cur: psycopg2 cursor
    :param requestdata: dictionary with request data
    :param tablename: name for the database table
    :param metaid: name of the id column for the metadata table
    :param filepath: full path to shapefile
    :param debug: True to print out info
    :return: None
    """
    print("preparing shapefile data upload")
    i=0

    #the function gey the name of the zipfile, but the unzipped shapefile may have a different name
    dirt = os.path.dirname(filepath)
    files = os.listdir(dirt)
    file = [i.split(".")[0] for i in files if i.endswith("shp")]

    sf = shapefile.Reader(dirt + "/" + file[0])
    #read the column names, we'll use this to get the column index
    # deletionflag is added and must be avoided
    fields = [field[0].strip().lower() for field in sf.fields if field[0].lower()!="deletionflag"]

    ### get id for current dataset
    datasetid = parse.unquote(requestdata["datasetid"])
    ### get column name index for current shapefile
    idvalue1 = fields.index(parse.unquote(requestdata["value1"]))

    # IMPORTANT   the code below works if the client sends the values in order!
    nvalues = 0
    while True:
        if requestdata.get("value2"):
            idvalue2 = fields.index(parse.unquote(requestdata['value2']));nvalues += 1
        else:break
        if requestdata.get("value3"):
            idvalue3 = fields.index(parse.unquote(requestdata['value3']));nvalues += 1
        else:break
        if requestdata.get("value4"):
            idvalue4 = fields.index(parse.unquote(requestdata['value4']));nvalues += 1
        else:break
        if requestdata.get("value5"):
            idvalue5 = fields.index(parse.unquote(requestdata['value5']));nvalues += 1
        else:break
        if requestdata.get("value6"):
            idvalue6 = fields.index(parse.unquote(requestdata['value6']));nvalues += 1
        break


    # prepare the statement to speed up the insertion
    print("updating field valuemap and prepare upload statements")
    update_valuemap(cur, requestdata,metaid,nvalues)
    prepare_upload_statements(requestdata, cur, tablename, metaid)

    print("uploading shapefile data")

    if nvalues == 0:
        if requestdata.get("row"):
            idrow = fields.index(parse.unquote(requestdata["row"]))
            for feature in sf.shapeRecords():
                #print(datasetid, feature.record[idrow], feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue])
                cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s)""",
                            (datasetid, feature.record[idrow], feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue1]))

                if debug:
                    i += 1
                    if i % 5000 == 0:
                        print(i)
        else:
            for feature in sf.shapeRecords():
                #print(datasetid, feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue])
                cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s)""",
                            (datasetid, feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue1]))

                if debug:
                    i += 1
                    if i % 5000 == 0:
                        print(i)

    elif nvalues == 1:
        if requestdata.get("row"):
            idrow = fields.index(parse.unquote(requestdata["row"]))
            for feature in sf.shapeRecords():
                # print(datasetid, feature.record[idrow], feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue])
                cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s)""",
                            (datasetid, feature.record[idrow], feature.shape.points[0][1], feature.shape.points[0][0],
                             feature.record[idvalue1],feature.record[idvalue2]))

        else:
            for feature in sf.shapeRecords():
                # print(datasetid, feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue])
                cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s)""",
                            (datasetid, feature.shape.points[0][1], feature.shape.points[0][0],
                             feature.record[idvalue1],feature.record[idvalue2]))

    elif nvalues == 2:
        if requestdata.get("row"):
            idrow = fields.index(parse.unquote(requestdata["row"]))
            for feature in sf.shapeRecords():
                # print(datasetid, feature.record[idrow], feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue])
                cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s, %s)""",
                            (datasetid, feature.record[idrow], feature.shape.points[0][1], feature.shape.points[0][0],
                             feature.record[idvalue1],feature.record[idvalue2],feature.record[idvalue3] ))

        else:
            for feature in sf.shapeRecords():
                # print(datasetid, feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue])
                cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s)""",
                            (datasetid, feature.shape.points[0][1], feature.shape.points[0][0],
                             feature.record[idvalue1],feature.record[idvalue2],feature.record[idvalue3]))

    elif nvalues == 3:
        if requestdata.get("row"):
            idrow = fields.index(parse.unquote(requestdata["row"]))
            for feature in sf.shapeRecords():
                # print(datasetid, feature.record[idrow], feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue])
                cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s, %s, %s)""",
                            (datasetid, feature.record[idrow], feature.shape.points[0][1], feature.shape.points[0][0],
                             feature.record[idvalue1],feature.record[idvalue2],feature.record[idvalue3],feature.record[idvalue4] ))

        else:
            for feature in sf.shapeRecords():
                # print(datasetid, feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue])
                cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s, %s)""",
                            (datasetid, feature.shape.points[0][1], feature.shape.points[0][0],
                             feature.record[idvalue1],feature.record[idvalue2],feature.record[idvalue3],feature.record[idvalue4]))

    elif nvalues == 4:
        if requestdata.get("row"):
            idrow = fields.index(parse.unquote(requestdata["row"]))
            for feature in sf.shapeRecords():
                # print(datasetid, feature.record[idrow], feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue])
                cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s, %s, %s, %s)""",
                            (datasetid, feature.record[idrow], feature.shape.points[0][1], feature.shape.points[0][0],
                             feature.record[idvalue1],feature.record[idvalue2],feature.record[idvalue3],feature.record[idvalue4],feature.record[idvalue5]  ))

        else:
            for feature in sf.shapeRecords():
                # print(datasetid, feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue])
                cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s, %s, %s)""",
                            (datasetid, feature.shape.points[0][1], feature.shape.points[0][0],
                             feature.record[idvalue1],feature.record[idvalue2],feature.record[idvalue3],feature.record[idvalue4],feature.record[idvalue5] ))

    elif nvalues == 5:
        if requestdata.get("row"):
            idrow = fields.index(parse.unquote(requestdata["row"]))
            for feature in sf.shapeRecords():
                # print(datasetid, feature.record[idrow], feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue])
                cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s, %s, %s, %s, %s)""",
                            (datasetid, feature.record[idrow], feature.shape.points[0][1], feature.shape.points[0][0],
                             feature.record[idvalue1],feature.record[idvalue2],feature.record[idvalue3],feature.record[idvalue4],feature.record[idvalue5],feature.record[idvalue6]))

        else:
            for feature in sf.shapeRecords():
                # print(datasetid, feature.shape.points[0][1], feature.shape.points[0][0], feature.record[idvalue])
                cur.execute("""EXECUTE  """ + tablename + """(%s,%s, %s,%s, %s, %s, %s, %s, %s)""",
                            (datasetid, feature.shape.points[0][1], feature.shape.points[0][0],
                             feature.record[idvalue1],feature.record[idvalue2],feature.record[idvalue3],feature.record[idvalue4],feature.record[idvalue5],feature.record[idvalue6]))



#todo add query for colordata, dictionary request may add  sensor:"colorgrade|berrysize|berrycount"
def get_geojson(requestdata,METADATA_DATA_TABLES, METADATA_IDS, DATA_IDS, limit=100000, proj=4326, conndict=None, cur=None):
    """
    Return the geojson for data. requestdata
    :param requestdata:  dictionary with request parameters

            for example  {"metatable":"canopy", "iddataset": "1"}

    :param METADATA_DATA_TABLES: mapping between metadata tables names and  dataset table names
    :param DATA_IDS: id fields names for data tables
    :param limit: the max number of features returned by the query, default 100000
    :param proj:
    :param conndict: dictionary with connection properties, if None the default connection will be used
    :param cur: psycopg2 cursor
    :return: the geojson
    """
    cur = cur
    conn = None
    hadcursor = True if cur else False  # set if we are using an already existing connection

    tablename = METADATA_DATA_TABLES[parse.unquote(requestdata.get('metatable'))]   # e.g. canopy->canodatum
    metaid = METADATA_IDS[parse.unquote(requestdata.get('metatable'))]   # e.g. canopy->id_canopy
    dataid = DATA_IDS[parse.unquote(requestdata.get('metatable'))]   # e.g.  canopy: "id_cdata"

    try:
        if not cur:
            if not conndict:
                conndict = default_connection
            conn = pgconnect(**conndict)
            cur = conn.cursor()

        cur.execute("""SELECT row_to_json(fc) FROM 
          ( SELECT 'FeatureCollection' As type, array_to_json(array_agg(f)) As features
          FROM (SELECT 'Feature' As type , ST_AsGeoJSON(lg.geom)::json As geometry, row_to_json(lp) As properties
           FROM %s As lg  INNER JOIN (SELECT %s, value FROM canodatum) As lp   #todo 'value FROM canodatum' should be changed!
               ON lg.%s = lp.%s where lg.%s=1 limit %s) As f)  As fc;""", #added limit otherwise query would run forever
                    (AsIs(tablename), AsIs(dataid), AsIs(dataid), AsIs(dataid),AsIs(metaid),limit ))# AsIs will hide the quotes

        return cur.fetchone()[0]

    except Exception as e:
        print(e)
        raise Exception("error querying database, try again")

    finally:
        if not hadcursor:  # if we passed the cursor we will not close the connection in this function
            if cur:
                cur.close()
            if conn:
                conn.close()




if __name__ == "__main__":

    UPLOAD_ROOT = "/vagrant/code/pysdss/data/input/uploads/"

    METADATA_ID_TYPES = {"yield": "id_ytype", "soil": "id_stype", "canopy": "id_ctype", "sensor": "id_stype"}
    METADATA_FIELDS_TYPES = {"datetime": "string", "roworientation": "string", "swathWidth": "number",
                             "ofset": "number", "rowSpacing": "number", "comments": "string"}
    METADATA_IDS = {"yield": "id_yield", "soil": "id_soil", "canopy": "id_canopy", "sensor": "id_sensor"}

    # link between metadata tables names and  dataset table names
    METADATA_DATA_TABLES = {"yield": "yielddatum", "soil": "soildatum", "canopy": "canodatum",
                            "sensor": "sensodatum", "sensodatum": ["berrycount", "berrysize", "colorgrade"]}

    # id fields for data tables
    DATA_IDS = {"yield": "id_ydata", "soil": "id_sdata", "canopy": "id_cdata", "sensor": "id_sdata"}
    # id fields for sensor subtables
    SENSOR_IDS = {"colorgrade": "id_cgrade", "berrysize": "id_size", "berrycount": "id_count"}


    def tests():
        ##tests####
        set_record('data.sensotype', "type", "remote sensing", conndict=default_connection)
        print(get_records('data.sensotype',("id_stype","type"), conndict=default_connection))
        delete_record('data.sensotype', "id_stype",18, conndict=default_connection)
    #tests()

    def test_upload_metadata():
        requestdata = {"metatable":"canopy", "toolid":"1", "datetime": "2017-08-15 14:19:25.63", "roworientation":"NE", "swathWidth":"90", "ofset":"5", "rowSpacing":"3", "comments":"hasta la vista"}

        print(upload_metadata(requestdata, METADATA_ID_TYPES, METADATA_FIELDS_TYPES,METADATA_IDS, conndict=None, cur=None))

    #test_upload_metadata()

    def test_upload_csv():
        requestdata = {"metatable":"canopy","datasetid": "2", "lat": "lat", "lon": "lng", "value1":"sf01","value2":"sf02","value3":"sf03","value4":"sf04","value5":"course","value6":"speed",
                       "row": "sensor_add", "swathwidth": "90", "ofset": "5", "rowSpacing": "3",
                       "comments": "hasta la vista"}

        conndict = default_connection
        conn = pgconnect(**conndict)
        cur = conn.cursor()

        tablename = "canodatum"
        metaid = "id_canopy"
        filepath = "/vagrant/code/pysdss/data/input/uploads/a5f9e0915ecb94449b26a8dc52b970cc0/2017-07-25 To Kalon NDVI.csv"

        upload_data_csv(cur, requestdata, tablename, metaid, filepath, debug=False)
        ###conn.commit()
        ###cur.close()
        ###conn.close()


    #test_upload_csv()

    def test_upload_shape():
        requestdata = {"metatable":"canopy","datasetid": "2", "lat": "lat", "lon": "lng", "value1":"sf01", "value2":"sf02","value3":"sf03", "value4":"sf04", "value5":"course", "value6":"speed",
                       "row": "sensor_add", "swathWidth": "90", "ofset": "5", "rowSpacing": "3",
                       "comments": "hasta la vista"}

        conndict = default_connection
        conn = pgconnect(**conndict)
        cur = conn.cursor()

        tablename = "canodatum"
        metaid = "id_canopy"
        filepath = "/vagrant/code/pysdss/data/input/uploads/a5f9e0915ecb94449b26a8dc52b970cc0/to_kalon.shp"
        upload_data_shape(cur, requestdata, tablename, metaid, filepath, debug=False)

        ###conn.commit()
        ###cur.close()
        ###conn.close()

    #test_upload_shape()


    def test_upload_data():

        ###filepath = "/vagrant/code/pysdss/data/input/uploads/a5f9e0915ecb94449b26a8dc52b970cc0/2017-07-25 To Kalon NDVI.csv"

        requestdata = {"metatable":"canopy", "filename":"2017-07-25 To Kalon NDVI.csv", "folderid":"a5f9e0915ecb94449b26a8dc52b970cc0", "datasetid": "2", "lat": "lat", "lon": "lng",
                       "value1":"sf01","value2":"sf02","value3":"sf03", "value4":"sf04", "value5":"course","value6":"speed",
                        "swathWidth": "90", "ofset": "5", "rowSpacing": "3","comments": "hasta la vistasss"}

        conndict = default_connection
        conn = pgconnect(**conndict)
        cur = conn.cursor()

        ######## upload csv
        #requestdata = {"metatable":"canopy", "filename":"2017-07-25 To Kalon NDVI.csv", "folderid":"a5f9e0915ecb94449b26a8dc52b970cc0", "datasetid": "1", "lat": "lat", "lon": "lng", "value":"sf01",
                       #"row": "sensor_addr", "swathWidth": "90", "ofset": "5", "rowSpacing": "3","comments": "hasta la vista"}
        #requestdata["row"] = "sensor_addr"
        requestdata["filename"] = "2017-07-25 To Kalon NDVI.csv"
        upload_data(requestdata, METADATA_DATA_TABLES, METADATA_IDS, UPLOAD_ROOT, proj=4326, conndict=None, cur=None)

        ####### upload shapefile
        #requestdata = {"metatable":"canopy", "filename":"", "folderid":"a5f9e0915ecb94449b26a8dc52b970cc0", "datasetid": "1", "lat": "lat", "lon": "lng", "value":"sf01",
                       #"row": "", "swathWidth": "90", "ofset": "5", "rowSpacing": "3","comments": "hasta la vista"}
        #requestdata["row"] = "sensor_add"
        requestdata["filename"] = "to_kalon.zip"#note the zip
        upload_data(requestdata, METADATA_DATA_TABLES, METADATA_IDS, UPLOAD_ROOT, proj=4326, conndict=None, cur=None)


    ##test_upload_data()


    def test_geojson():
        requestdata ={"metatable": "canopy", "datasetid": "1"}
        result = get_geojson(requestdata, METADATA_DATA_TABLES, METADATA_IDS, DATA_IDS, limit=100000,proj=4326, conndict=None, cur=None)
        result = str(result)
        #print(result)
        g=open(UPLOAD_ROOT+"/xxx.json",'w')
        g.write(str(result))

    #test_geojson()


    def test_valuemap():

        requestdata = {"metatable": "sensor", "datasetid": "44", "value1":"col1","value2":"col2","value3":"col3","value4":"col4","value5":"col5","value6":"col6" }

        conndict = default_connection
        conn = pgconnect(**conndict)
        cur = conn.cursor()

        #for i in [1,2,3,4,5]:
        #    jsn = update_valuemap(cur,requestdata,nvalues=i)
        #    print(jsn)
        #jsn = update_valuemap(cur, requestdata, nvalues=5)
        ###conn.commit()
        cur.close()
        conn.close()

    #test_valuemap()
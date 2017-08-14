# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        database.py
# Purpose:     upload data to postgresql database
#
# Author:      claudio piccinini
#
# Updated:     08/02/2017
#-------------------------------------------------------------------------------

#TODO the csv has Nan for missing numerical values. Empy values would create an empty string wich would cause a psycopg error
#TODO i should create a script to change the NaN in the database with null
#TODO some sql receive user parameters, danger of sql injection?
#TODO what about concurrent use???
# TODO what about the transformed values in the colorgrade table?
#TODO the processes should log the location of the output files
#todo fix code for database dataset filter functions (look at line 747)

import csv
import os
import uuid
import datetime
import json
import psycopg2
from psycopg2.extensions import AsIs
import numpy as np
import pandas as pd

import pysdss.utils as utils

from pysdss.database import default_connection
from pysdss.database import pgconnect
#default_connection = {"dbname": "grapes", "user": "claudio", "password": "claudio", "port":"5432", "host": "127.0.0.1"}

# class dataToPostgres():
#   raise NotImplemented

# enumeration for geoprocessing messages
from enum import Enum
geomessages = Enum ("geomessages", "starting running stopped error completed unauthorized")

# enumeration for geoprocessing types
geotypes = Enum ("geotypes", "statistics filter_bystd filter_byvalue filter_bystep kriging interpolation")


def csvToPostgres(conndict, stypeid, filename, columns, proj=4326, debug=True):
    # TODO check the columns parameter for the mapping, the user should be shown a form to couple csv columns with database columns

    #TODO: should select also the date of the data?the user should select the sensor type when uploading the data
    """ upload a csv sensor file to postgresql
    :param conndict: dictionary with postgresql connection
            e.g. {"dbname": "grapes", "user": "claudio", "password": "claudio", "port":"5432", "host": "127.0.0.1"}
    :param stypeid: the id of the sensor type
    :param filename: the path to the uploaded file
    :param columns: mapping between csv columns and database columns
    :param proj: EPSG code for storing the geometry
    :param debug: use this to print messages
    :return: "" if everything is OK else return error message string
    """
    ##typ = "proximal sensing"
    #proj = 4326

    # read csv, filter columns
    #I did not use postgresql COPY because of the foreign keys #TODO: could I use COPY?

    #TODO check if this is ok in production code, probably filename should have the absolute path
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    conn=""
    cur=""
    f=""

    try:
        # TODO check if this is ok in production code, probably filename should have the absolute path
        f = open(os.getcwd()+"/data/"+filename)
    except:
        raise IOError("Cannot open the file")
    else:

        reader = csv.reader(f)
        #get the filed names, and strio/lower them, i did not use dictreader because of the csv table with strip problems
        columns = reader.__next__()
        columns = [a.strip().lower() for a in columns]

        #TODO mapuser columns to database columns

        # connect to the database, iterate csv, and upload data
        try:
            print("connecting to database")
            # http: // stackoverflow.com / questions / 17443379 / psql - fatal - peer - authentication - failed -for -user - dev
            conn = pgconnect(**conndict)

            cur = conn.cursor()
            conn.autocommit = False

            first_line = True
            ID_sensor = ""

            print("preparing execute statements")
            #prepare the execute statements to speed up the insert (did not use copy because of the foreign keys!)
            cur.execute("""PREPARE sensodatum(integer,smallint, double precision, double precision) AS INSERT INTO data.SensoDatum (ID_sensor,row, lat,lon)
            VALUES ($1,$2,$3, $4) RETURNING ID_sdata""")
            cur.execute("""PREPARE colorgrade(integer,real,real,real,real) AS INSERT INTO data.ColorGrade (ID_sdata, A,B,C,D )
                    VALUES ($1,$2,$3, $4, $5)""")
            cur.execute("""PREPARE berrysize(integer,real) AS INSERT INTO data.BerrySize (ID_sdata, value ) VALUES ($1,$2)""")
            cur.execute("""PREPARE berrycount(integer,smallint) AS INSERT INTO data.BerryCount (ID_sdata, value ) VALUES ($1,$2)""")

            i = 0
            for line in reader:

                    if first_line: #we need to set 1 row in the Sensor table with the dataset metadata

                        ''' this is not necessary anymore as the function gets the sensor type id as parameter
                        #get the id for the sensor type(user should select the sensor type when uploading the data)
                        cur.execute("""SELECT id_stype from data.SensoType where type = %s""", (typ,))
                        id_stype = cur.fetchone()[0]
                        #insert one row into the Sensor table
                        cur.execute("""INSERT INTO data.Sensor (ID_stype,ID_dataset) VALUES (%s,%s) RETURNING ID_sensor""",
                                    (id_stype, line[columns.index('%dataset_id')])) #here missing all the other values
                        '''

                        # insert one row into the Sensor table
                        cur.execute(
                            """INSERT INTO data.Sensor (ID_stype,ID_dataset) VALUES (%s,%s) RETURNING ID_sensor""",
                            (stypeid, line[columns.index('%dataset_id')]))  # here missing all the other values

                        ID_sensor = cur.fetchone()[0]

                    first_line = False #after first line we skip the above

                    cur.execute("""EXECUTE  sensodatum(%s,%s, %s,%s)""",
                                (ID_sensor,line[columns.index('row')],line[columns.index('latitude')], line[columns.index('longitude')]))
                    ID_sdata = cur.fetchone()[0]

                    cur.execute("""EXECUTE  colorgrade(%s,%s,%s,%s,%s)""", (ID_sdata,
                                  line[columns.index('harvestable_a')], line[columns.index('harvestable_b')],
                                  line[columns.index('harvestable_c')], line[columns.index('harvestable_d')]))
                    cur.execute("""EXECUTE  berrysize(%s,%s)""", (ID_sdata,line[columns.index('fruit_diameter_m')]))
                    cur.execute("""EXECUTE  berrycount(%s,%s)""", (ID_sdata,line[columns.index('raw_fruit_count')]))

                    ##TODO insert the images?

                    if debug:
                        i+=1
                        if i%5000==0:
                            print(i)

        except Exception as e:
            print(e)
            if cur:
                cur.close()
            if conn:
                conn.rollback()
                conn.close()
            if f:
                f.close()
            raise e

        else:
            print( "total rows: "+str(i))
            conn.commit()

            try:
                conn.autocommit = True #necessary for vacuum
                print("creating geometry")
                ## fill geometry column
                cur.execute("""UPDATE data.SensoDatum SET geom = ST_PointFromText('POINT(' || lon || ' ' || lat || ')', %s)""", (proj,))
                print("creating index")
                ## create geo index
                try: #create spatial index if it does not exists
                    cur.execute("""CREATE INDEX idx_sensodatum_geom ON data.SensoDatum USING GIST (geom)""")
                except Exception as e:
                    print(e) #this will happen if the spatial index is already there

                print("vacuum analize clustering")
                cur.execute("""VACUUM ANALYZE data.SensoDatum""")
                cur.execute("""VACUUM ANALYZE data.Sensor""")
                cur.execute("""VACUUM ANALYZE data.ColorGrade""")
                cur.execute("""VACUUM ANALYZE data.BerryCount""")
                cur.execute("""VACUUM ANALYZE data.BerrySize""")
                cur.execute("""CLUSTER data.SensoDatum USING idx_sensodatum_geom""")
                cur.execute("""ANALYZE data.SensoDatum""")

            except Exception as e:
                print(e)
                raise e

            finally:
                if cur:
                    cur.close()
                if conn:
                    conn.close()

    finally:
        if f:
            f.close()

def shapeToPostgres():
    raise NotImplementedError

def filter_bystd(conndict, datasetid, colname="A", nstd=1, userows = False, boundid=None, outpath="/vagrant/code/STARS/berry/data/", export=True, debug=True):
    """
    Reduce the number of sensor points using an average and standard deviation, reduction is based on a
    specific colorgrade field. Points with value 0 are not part of the result
    Average and std can be global or calculated for each row.
    The user can choose to restrict the filter to a polygon boundary.
    the function set the 'keep' fields to True for those points that we want to keep,
    result can be saved as a csv on disk (recommended in a multiuser database!!!)

    :param conndict: dictionary with postgresql connection
            e.g. {"dbname": "grapes", "user": "claudio", "password": "claudio", "port":"5432", "host": "127.0.0.1"}
    :param datasetid: the dataset id to filter
    :param colname: the colorgrade column to filter
    :param nstd: the number of standard deviation for the filter
    :param userows: if True calculate the statistics for each row
    :param boundid: the polygon boundary id
    :param outpath: the path to the result folder
    :param export: True to export the result to csv (recommended True in a multiuser database!!!)
    :param debug: print debug messages on the console
    :return:
    """

    #create unique name for temporary table
    tblname = utils.create_uniqueid()
    conn=""
    cur=""
    outpath = outpath + "/std_byrow" if userows else outpath + "/std"


    try:
        conn = pgconnect(**conndict)
        cur = conn.cursor()
        #conn.autocommit = False

        #create temporary table for a dataset and specific color grade where colorgrade!=0
        # TODO check if the join data.sensor and data.SensoDatum is necessary (solved 15/02)

        '''cur.execute("""     CREATE TEMP TABLE %s ON COMMIT DROP AS(
                            SELECT data.SensoDatum.row as row,data.colorgrade.ID_cgrade,data.colorgrade.ID_sdata,
                            data.colorgrade.%s, data.colorgrade.t%s,  data.colorgrade.keep%s
                            FROM data.sensor
                            INNER JOIN
                            data.SensoDatum
                            ON data.sensor.id_sensor = data.SensoDatum.id_sensor
                            INNER JOIN data.colorgrade
                            ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                            where data.sensor.id_sensor= %s AND data.colorgrade.%s!=0)""", (AsIs(tblname),AsIs(colname),AsIs(colname),AsIs(colname),datasetid, AsIs(colname)))
        '''

        corequery = """     SELECT data.SensoDatum.row, data.SensoDatum.geom, data.colorgrade.ID_cgrade,data.colorgrade.ID_sdata,
                            data.colorgrade.%s, data.colorgrade.t%s, data.colorgrade.keep%s
                            --FROM data.sensor
                            --INNER JOIN
                            FROM data.SensoDatum
                            --ON data.sensor.id_sensor = data.SensoDatum.id_sensor
                            INNER JOIN data.colorgrade
                            ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                            where data.SensoDatum.id_sensor= %s AND data.colorgrade.%s!=0"""


        if boundid: #if we choose  polytgon boundary we need to select the points inside
            cur.execute("""     CREATE TEMP TABLE %s ON COMMIT DROP AS(
                                select foo.row, foo.geom, foo.ID_cgrade,foo.ID_sdata,
                                foo.%s, foo.t%s, foo.keep%s
                                from ("""+corequery+""") as foo
                                INNER JOIN data.boundary
                                ON ST_Contains(boundary.geom, foo.geom)
                                where boundary.id_bound = %s
                                )
                                """, (AsIs(tblname), AsIs(colname),AsIs(colname),AsIs(colname), AsIs(colname),AsIs(colname),AsIs(colname),datasetid, AsIs(colname),boundid))
        else:
            cur.execute("""     CREATE TEMP TABLE %s ON COMMIT DROP AS(""" + corequery + """)""",
                        (AsIs(tblname), AsIs(colname), AsIs(colname), AsIs(colname), datasetid, AsIs(colname)))

        #initialize the field keep to FALSE
        cur.execute("""update %s set keep%s=FALSE""", (AsIs(tblname),AsIs(colname))) #TODO check if this is necessary

        #initialize the field keep to FALSE for the dataset
        cur.execute("""update data.colorgrade
                        set keep%s = FALSE
                        from (select id_cgrade from %s) as x
                        WHERE data.colorgrade.id_cgrade = x.id_cgrade""",(AsIs(colname), AsIs(tblname),))

        if userows: #calculate range for each row and filter for each row


            cur.execute("""select row, avg(%s), stddev(%s) from %s
                            group by row
                            order by row""", (AsIs(colname), AsIs(colname), AsIs(tblname)))

            result = cur.fetchall()

            for s in result:
                mn = s[1] - (s[2] * nstd)
                mx = s[1] + (s[2] * nstd)

                if debug: print(s[0],s[1],s[2],mn,mx )

                cur.execute("""update data.colorgrade
                                set keep%s = TRUE
                                from (select id_cgrade,row from %s where row=%s and %s >%s and %s <%s  ) as x
                                WHERE data.colorgrade.id_cgrade = x.id_cgrade""",(AsIs(colname),  AsIs(tblname),s[0], AsIs(colname),mn, AsIs(colname),mx))

        else: #use statistics from the entire dataset

            # get average and standard deviation and calculate the range
            cur.execute("""select avg(%s), stddev(%s) from %s""", (AsIs(colname), AsIs(colname), AsIs(tblname)))
            avg, std = cur.fetchone()

            if debug: print(avg, std)

            mn = avg - (std * nstd)
            mx = avg + (std * nstd)

            cur.execute("""update data.colorgrade
                            set keep%s = TRUE
                            from (select id_cgrade from %s where %s >%s and %s <%s ) as x
                            WHERE data.colorgrade.id_cgrade = x.id_cgrade""",(AsIs(colname),AsIs(tblname), AsIs(colname), mn, AsIs(colname), mx))



        #TODO test if this exporting is working properly (multiaccess problems)
        if export:
            if boundid:
                #i am passing the cursor so that I can stay in the same transaction (hopefully!!!)
                download_colordata(datasetid, colname=colname, outfile=outpath, boundid=boundid, filtered=True, cur=cur)
            else:
                download_colordata(datasetid, colname=colname, outfile=outpath, filtered=True, cur=cur)

        # here the temporary table will be deleted because on commit drop
        conn.commit()

    except Exception as e:
        print(e)
        if conn:
            conn.rollback()
        raise(e)


    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def get_statistics(conndict, datasetid, colname="A",userows = False,boundid=None, debug=True):
    """
    Get statistics for the entire dataset and specific colorgrade column
    :param conndict: dictionary with postgresql connection
            e.g. {"dbname": "grapes", "user": "claudio", "password": "claudio", "port":"5432", "host": "127.0.0.1"}
    :param datasetid:
    :param colname:
    :param userows:
    :param boundid: the polygon boundary id
    :param debug:
    :return:
    """

    try:
        conn = pgconnect(**conndict)
        cur = conn.cursor()


        corequery = """SELECT data.SensoDatum.row, data.SensoDatum.geom, data.colorgrade.%s
                        --FROM data.sensor
                        --INNER JOIN
                        FROM data.SensoDatum
                        --ON data.sensor.id_sensor = data.SensoDatum.id_sensor
                        INNER JOIN data.colorgrade
                        ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                        where data.SensoDatum.id_sensor= %s AND data.colorgrade.%s!=0"""



        if boundid: #if we choose  polytgon boundary we need to select the points inside
            if userows:
                cur.execute("""select row, avg(%s), stddev(%s) from (
                                    select foo.row, foo.%s from (
                                        select k.row, k.%s from(""" +corequery+""") as k
                                        INNER JOIN
                                        data.boundary ON ST_Contains(boundary.geom, k.geom)
                                        where boundary.id_bound = %s
                                    ) As foo
                                ) as foo2
                                group by row
                                order by row
                                """,(AsIs(colname),AsIs(colname),AsIs(colname),AsIs(colname), AsIs(colname),datasetid, AsIs(colname), boundid))

                result = cur.fetchall()
                if debug:
                    for i in result:
                        print(i)
                return result

            else:
                cur.execute("""     select avg(%s), stddev(%s) from (
                                    select k.row, k.%s from(""" +corequery+""") as k
                                    INNER JOIN
                                    data.boundary ON ST_Contains(boundary.geom, k.geom)
                                    where boundary.id_bound = %s ) AS foo
                                    """,(AsIs(colname),AsIs(colname),AsIs(colname), AsIs(colname),datasetid, AsIs(colname), boundid))

                avg, std = cur.fetchone()
                if debug:
                    print(avg, std)
                return avg, std

        else:  # no boundary

            if userows:
                cur.execute("""select row, avg(%s), stddev(%s) from
                                (""" +corequery+""") As foo
                                group by row
                                order by row""", (AsIs(colname),AsIs(colname), AsIs(colname),datasetid, AsIs(colname)))

                result = cur.fetchall()
                if debug:
                    for i in result:
                        print(i)
                return result
            else:

                cur.execute("""     select avg(%s), stddev(%s) from (
                                    """+corequery+"""
                                    ) AS foo""",
                                    (AsIs(colname),AsIs(colname),  AsIs(colname), datasetid, AsIs(colname)))

            avg, std = cur.fetchone()
            if debug:
                print(avg, std)
            return avg, std

    except Exception as e:
        print(e)
        raise (e)

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def download_colordata( datasetid, colname="A", allcolors=False, boundid=None, outfile="/vagrant/code/STARS/berry/data/data", filtered=False, conndict=None, cur=None):
    """
    Dowload colordata for a specific dataset and column; can be all data ina single file, or splitted in 2 files by the keep field
    The user can pass a psycopg2 cursor to use an already existing connection
    :param datasetid:
    :param colname:
    :param allcolors
    :param boundid if we want to clip the result to a polygon
    :param outfile:
    :param filtered:  output 2 files, one for the filtered and another one for non filtered
    :param conndict: dictionary with postgresql connection
            e.g. {"dbname": "grapes", "user": "claudio", "password": "claudio", "port":"5432", "host": "127.0.0.1"}
    :param cur: psycopg2 cursor
    :return:
    """

    fkeep = ""
    fdelete = ""
    f = ""
    hadcursor = True if cur else False #set if we are using an already existing connection

    try:

        if not cur:
            conn = pgconnect(**conndict)
            cur = conn.cursor()

        if allcolors: #want all colors

            str1=""" data.colorgrade.A, data.colorgrade.B,data.colorgrade.C,data.colorgrade.D """
            str2= """ AND data.colorgrade.keepA=TRUE AND data.colorgrade.keepB=TRUE
            AND data.colorgrade.keepC=TRUE AND data.colorgrade.keepD=TRUE """
            str3= """ AND data.colorgrade.keepA=FALSE AND data.colorgrade.keepB=FALSE
            AND data.colorgrade.keepC=FALSE AND data.colorgrade.keepD=FALSE """

        else:
            str1 = """ data.colorgrade.""" + colname
            str2= """ AND data.colorgrade.keep"""+colname+"""=TRUE """
            str3 = """ AND data.colorgrade.keep""" + colname + """=FALSE """




        if boundid:
            row1 = """SELECT data.SensoDatum.row, data.SensoDatum.id_sdata, data.sensodatum.lat, data.sensodatum.lon, data.sensodatum.geom, """
        else:
            row1 = """SELECT data.SensoDatum.row, data.SensoDatum.id_sdata, data.sensodatum.lat, data.sensodatum.lon, """

        corequery = row1+str1+""" FROM data.SensoDatum
                                    INNER JOIN data.colorgrade
                                    ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                                    where data.SensoDatum.id_sensor=""" + str(datasetid)

        if boundid:

            querystart = """SELECT k.row, k.id_sdata, k.lat, k.lon, k.""" + colname
            if allcolors:
                colname = "ABCD"
                querystart = """SELECT k.row, k.id_sdata, k.lat, k.lon,
                                k.A, k.B,k.C,k.D"""

            if filtered:  # output 2 files
                fkeep = open(outfile + "_"+colname+ "_boundary_"+str(boundid)+"_keep.csv", "w")
                cur.copy_expert("""COPY (
                                """+ querystart +
                                """ from("""+corequery+ str2 + """) AS k
                                INNER JOIN data.boundary ON ST_Contains(boundary.geom, k.geom)
                                where boundary.id_bound =""" + str(boundid) + """
                                order by row, id_sdata
                                ) TO STDOUT WITH CSV HEADER""", fkeep)

                fdelete = open(outfile + "_" + colname + "_boundary_"+str(boundid)+ "_delete.csv", "w")
                cur.copy_expert("""COPY (
                                """+ querystart +
                                """ from("""+corequery+ str3 + """) AS k
                                INNER JOIN data.boundary ON ST_Contains(boundary.geom, k.geom)
                                where boundary.id_bound = """ + str(boundid) + """
                                order by row, id_sdata
                                ) TO STDOUT WITH CSV HEADER""", fdelete)

            else:
                f = open(outfile + "_"+ colname + "_boundary_"+str(boundid) + ".csv", "w")
                cur.copy_expert("""COPY (
                                """
                                + querystart +
                                """ from("""+corequery+""") AS k
                                INNER JOIN data.boundary ON ST_Contains(boundary.geom, k.geom)
                                where boundary.id_bound = 1
                                order by row, id_sdata
                                ) TO STDOUT WITH CSV HEADER""", f)

        else: #no bounding
            if allcolors: colname = "ABCD"
            if filtered:  # output 2 files
                fkeep = open(outfile + "_"+colname+ "_keep.csv", "w")
                cur.copy_expert("""COPY (

                                SELECT data.SensoDatum.row as row, data.SensoDatum.id_sdata as id_sdata, data.sensodatum.lat, data.sensodatum.lon,"""+str1+"""
                                 FROM data.sensor
                                INNER JOIN
                                data.SensoDatum
                                ON data.sensor.id_sensor = data.SensoDatum.id_sensor
                                INNER JOIN data.colorgrade
                                ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                                where data.sensor.id_sensor=""" + str(datasetid)

                                + str2 + """ order by row, id_sdata

                                ) TO STDOUT WITH CSV HEADER""", fkeep)

                fdelete = open(outfile + "_" + colname + "_delete.csv", "w")
                cur.copy_expert("""COPY (


                                SELECT data.SensoDatum.row as row, data.SensoDatum.id_sdata as id_sdata, data.sensodatum.lat, data.sensodatum.lon, """+str1+"""
                                 FROM data.sensor
                                INNER JOIN
                                data.SensoDatum
                                ON data.sensor.id_sensor = data.SensoDatum.id_sensor
                                INNER JOIN data.colorgrade
                                ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                                where data.sensor.id_sensor=""" + str(datasetid)

                                + str3 +""" order by row, id_sdata


                                ) TO STDOUT WITH CSV HEADER""", fdelete)

            else:
                f = open(outfile + "_"+ colname + ".csv", "w")
                cur.copy_expert("""COPY (


                                SELECT data.SensoDatum.row as row, data.SensoDatum.id_sdata as id_sdata, data.sensodatum.lat, data.sensodatum.lon, """+str1+"""
                                FROM data.sensor
                                INNER JOIN
                                data.SensoDatum
                                ON data.sensor.id_sensor = data.SensoDatum.id_sensor
                                INNER JOIN data.colorgrade
                                ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                                where data.sensor.id_sensor=""" + str(datasetid)

                                + """ order by row, id_sdata


                                ) TO STDOUT WITH CSV HEADER""", f)

    except Exception as e:
        print(e)
        raise (e)

    finally:

        if not hadcursor: #if we passed the cursor we will not close the connection in this function
            if cur:
                cur.close()
            if conn:
                conn.close()
        if fkeep:
            fkeep.close()
        if fdelete:
            fdelete.close()
        if f:
            f.close()


def download_data(leftid, lefttable, leftprkey, leftcolmns, rightid, righttable, rightcolmns, value, outfile, jointype="INNER JOIN",
                  leftboundgeom=None,boundtable=None, righttboundgeom=None, boundid=None, boundvalue=None, orderclms=None, conndict=None, cur=None, debug=False):
    """
    Download 2 joined database table data to a csv file
    a clipping polygon can be used (be sure that the left table has geometry!)

    # example of output query without clipping
    SELECT data.SensoDatum.row,data.SensoDatum.id_sdata,data.SensoDatum.lat,data.SensoDatum.lon,data.colorgrade.d
    ,data.colorgrade.keepd::int from data.SensoDatum
    INNER JOIN data.colorgrade
    ON data.SensoDatum.id_sdata=data.colorgrade.id_sdata
    where data.SensoDatum.id_sensor=26

    # example of output query with clipping
    select k.row,k.id_sdata,k.lat,k.lon,k.d,k.keepd::int from
    (SELECT data.SensoDatum.row,data.SensoDatum.id_sdata,data.SensoDatum.lat,data.SensoDatum.lon,data.colorgrade.d,
        data.colorgrade.keepd::int,data.SensoDatum.geom from data.SensoDatum
        INNER JOIN data.colorgrade
        ON data.SensoDatum.id_sdata=data.colorgrade.id_sdata
        where data.SensoDatum.id_sensor=26) AS k
    INNER JOIN data.boundary ON ST_Contains(data.boundary.geom, k.geom)
    where data.boundary.id_bound=1
    order by row,id_sdata

    :param leftid: left table join column name
    :param lefttable: left table name
    :param leftprkey: left table primary key name used for filter the selection
    :param leftcolmns: left table columns to return in the result
    :param rightid: right table join column name
    :param righttable: right table name
    :param rightcolmns: right table columns to return in the result
    :param value: left table primary key value
    :param outfile: absolute path for output file
    :param jointype: join type (default to inner join)
    :param leftboundgeom: column name for the boundtable
    :param boundtable: table with the polygon
    :param righttboundgeom: column name for the geometry
    :param boundid: column name for the primary key in the polygon table
    :param boundvalue: bound table primary key value
    :param orderclms: list with columns to order the result
    :param conndict: dictionary with postgresql connection, pass this to create a new connection
            e.g. {"dbname": "grapes", "user": "claudio", "password": "claudio", "port":"5432", "host": "127.0.0.1"}
    :param cur: psycopg2 cursor ; pass this if the connection already exists
    :param: debug: True to print console messages
    :return:
    """

    hadcursor = True if cur else False #set if we are using an already existing connection
    f=""

    try:

        if not cur:
            conn = pgconnect(**conndict)
            cur = conn.cursor()

        # set fields full path
        leftid = lefttable + "." + leftid
        rightid = righttable + "." + rightid
        leftprkey = lefttable + "." + leftprkey
        # join columns to build the search string
        str1 = ",".join([lefttable + "." + i for i in leftcolmns] + [righttable + "." + i for i in rightcolmns])

        if boundtable: # for clipping we need to define a subquery
            str1 =str1 + "," + lefttable + "." + righttboundgeom
            str2 = ",".join(["k." + i for i in leftcolmns] + ["k." + i for i in rightcolmns])
            corequery = """SELECT """+ str1 + " from " + lefttable + """ """ +  jointype + """ """ + righttable + """ ON """ + leftid \
            + """=""" + rightid + """ where """ + leftprkey + """=""" + str(value)
        else:
            corequery = """SELECT """+ str1 + " from " + lefttable + """ """ +  jointype + """ """ + righttable + """ ON """ + leftid \
            + """=""" + rightid + """ where """ + leftprkey + """=""" + str(value)

        ordr=""
        if orderclms: # add result ordering
            ordr = " order by " + (",".join(orderclms))

        f = open(outfile, "w")

        if boundtable:

            query = """COPY (
                                select """\
                                + str2 +\
                                """ from("""+corequery+""") AS k
                                INNER JOIN """+ boundtable +""" ON ST_Contains("""+ boundtable+"."+ leftboundgeom +""", k."""+ righttboundgeom+""")
                                where """+ boundtable+"."+ boundid +"""=""" + str(boundvalue)+ ordr + """
                                ) TO STDOUT WITH CSV HEADER"""

            if debug: print(query)
            cur.copy_expert(query,f)

        else:

            query = """COPY ( """+ corequery + ordr + """) TO STDOUT WITH CSV HEADER"""
            if debug: print(query)
            cur.copy_expert(query,f)

        return

    except Exception as e:
        print(e)
        raise (e)

    finally:

        if not hadcursor: #if we passed the cursor we will not close the connection in this function
            if cur:
                cur.close()
            if conn:
                conn.close()
        if f:
            f.close()


def filter_byvalue(conndict, datasetid, colname = "A",value=0.8,filterzero=True, operator=">=", boundid=None, outpath = "/vagrant/code/STARS/berry/data/", export = True, debug = True):
    """
    Reduce the number of sensor points using a threshold value and an operator (>= or <=). If '>=' result will
    keep the row if 'rowvalue' >= 'value', otherwise will keep 'rowvalue' <= 'value'
    reduction is based on a specific colorgrade field. Points with value 0 are not part of the result in 'filterzero' is True
    The user can choose to restrict the filter to a polygon boundary.
    the function set the 'keep' fields to True for those points that we want to keep,
    result can be saved as a csv on disk (recommended in a multiuser database!!!)

    :param conndict: dictionary with postgresql connection
            e.g. {"dbname": "grapes", "user": "claudio", "password": "claudio", "port":"5432", "host": "127.0.0.1"}
    :param datasetid: dataset id
    :param colname: field name
    :param value: threshold value
    :param filterzero: if True the value zero won't be keep
    :param operator: >= for keeping values >= value, otherwise keep values <=value
    :param outpath: the path to the output csv file
    :param export: True to export the result to csv (recommended in multiuser environment!!!)
    :param debug: print debug messages on the console
    :return:
    """
    #create unique name for temporary table, also add a front letter and delete - to get a correct table name
    tblname = utils.create_uniqueid()

    conn=""
    cur=""
    outpath = outpath + "/value_more_equal"+str(value) if operator==">=" else outpath + "/value_less_equal"+str(value)
    outpath = outpath +"_nozero" if filterzero else outpath

    try:
        conn = pgconnect(**conndict)
        cur = conn.cursor()
        #conn.autocommit = False

        #create temporary table for a dataset and specific color grade


        corequery = """SELECT data.SensoDatum.row, data.SensoDatum.geom, data.colorgrade.ID_cgrade,data.colorgrade.ID_sdata,
                        data.colorgrade.%s, data.colorgrade.t%s, data.colorgrade.keep%s
                        FROM data.SensoDatum
                        INNER JOIN data.colorgrade
                        ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                        where data.SensoDatum.id_sensor= %s """

        ''' old code
        if filterzero:
            cur.execute("""     CREATE TEMP TABLE %s ON COMMIT DROP AS(
                            SELECT data.colorgrade.ID_cgrade as ID_cgrade, data.colorgrade.%s as %s, data.colorgrade.t%s as t%s
                            FROM data.sensor
                            INNER JOIN
                            data.SensoDatum
                            ON data.sensor.id_sensor = data.SensoDatum.id_sensor
                            INNER JOIN data.colorgrade
                            ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                            where data.sensor.id_sensor= %s AND data.colorgrade.%s!=0)""", (AsIs(tblname),AsIs(colname),AsIs(colname),AsIs(colname),AsIs(colname),datasetid, AsIs(colname)))
        else:
            cur.execute("""     CREATE TEMP TABLE %s ON COMMIT DROP AS(
                            SELECT data.colorgrade.ID_cgrade as ID_cgrade, data.colorgrade.%s as %s, data.colorgrade.t%s as t%s
                            FROM data.sensor
                            INNER JOIN
                            data.SensoDatum
                            ON data.sensor.id_sensor = data.SensoDatum.id_sensor
                            INNER JOIN data.colorgrade
                            ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                            where data.sensor.id_sensor= %s)""", (AsIs(tblname),AsIs(colname),AsIs(colname),AsIs(colname),AsIs(colname),datasetid))
        '''
        #initialize the field keep to FALSE
        #cur.execute("""update %s set keep%s=FALSE""", (AsIs(tblname),AsIs(colname)))


        if filterzero:  #### need to add AND data.colorgrade. % s != 0 to the corequery
            if boundid:  # if we choose  polytgon boundary we need to select the points inside
                cur.execute("""CREATE TEMP TABLE %s ON COMMIT DROP AS(
                                    select foo.row, foo.geom, foo.ID_cgrade,foo.ID_sdata,
                                    foo.%s, foo.t%s, foo.keep%s
                                    from (""" + corequery + """ AND data.colorgrade.%s!=0) as foo
                                    INNER JOIN data.boundary
                                    ON ST_Contains(boundary.geom, foo.geom)
                                    where boundary.id_bound = %s
                                    )
                                    """, (
                AsIs(tblname), AsIs(colname), AsIs(colname), AsIs(colname), AsIs(colname), AsIs(colname), AsIs(colname),
                datasetid, AsIs(colname), boundid))
            else:
                cur.execute("""     CREATE TEMP TABLE %s ON COMMIT DROP AS(""" + corequery + """ AND data.colorgrade.%s!=0)""",
                            (AsIs(tblname), AsIs(colname), AsIs(colname), AsIs(colname), datasetid, AsIs(colname)))
        else:
            if boundid:  # if we choose  polytgon boundary we need to select the points inside
                cur.execute("""     CREATE TEMP TABLE %s ON COMMIT DROP AS(
                                    select foo.row, foo.geom, foo.ID_cgrade,foo.ID_sdata,
                                    foo.%s, foo.t%s, foo.keep%s
                                    from (""" + corequery + """) as foo
                                    INNER JOIN data.boundary
                                    ON ST_Contains(boundary.geom, foo.geom)
                                    where boundary.id_bound = %s
                                    )
                                    """, (
                AsIs(tblname), AsIs(colname), AsIs(colname), AsIs(colname), AsIs(colname), AsIs(colname), AsIs(colname),
                datasetid, boundid))
            else:
                cur.execute("""     CREATE TEMP TABLE %s ON COMMIT DROP AS(""" + corequery + """)""",
                            (AsIs(tblname), AsIs(colname), AsIs(colname), AsIs(colname), datasetid))

        #initialize the field keep to FALSE for the dataset
        ###the code below was wrong because the download_colordata function will download all the rows for the dataset
        #cur.execute("""update data.colorgrade
        #                set keep%s = FALSE
        #               from (select id_cgrade from %s) as x
        #                WHERE data.colorgrade.id_cgrade = x.id_cgrade""",(AsIs(colname), AsIs(tblname),))

        #set the keep column to False for this dataset (
        initialize_dataset_column(datasetid, colname="keep"+colname, value=False, cur=cur)


        if  (operator==">="):
            cur.execute("""update data.colorgrade
                            set keep%s = TRUE
                            from (select id_cgrade from %s where %s >= %s ) as x
                            WHERE data.colorgrade.id_cgrade = x.id_cgrade""",(AsIs(colname),AsIs(tblname), AsIs(colname), value))

        else: #TODO why is this putting the threshold value in the non keep part!!!
            cur.execute("""update data.colorgrade
                            set keep%s = TRUE
                            from (select id_cgrade from %s where %s <= %s ) as x
                            WHERE data.colorgrade.id_cgrade = x.id_cgrade""",(AsIs(colname),AsIs(tblname), AsIs(colname), value))

        #TODO test if this exporting is working properly (multiaccess problems)
        if export:
            if boundid:
                #i am passing the cursor so that I can stay in the same transaction (hopefully!!!)
                download_colordata(datasetid, colname=colname, outfile=outpath, boundid=boundid, filtered=True, cur=cur)
            else:
                download_colordata(datasetid, colname=colname, outfile=outpath, filtered=True, cur=cur)

        # here the temporary table will be deleted because on commit drop
        conn.commit()


    except Exception as e:
        print(e)
        if conn:
            conn.rollback()
        raise(e)

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def initialize_dataset_column(datasetid, colname="keepA", value=False, conndict=None, cur=None):
    """
    Set a boolean column in the colorgrade table to TRUE or FALSE for a specific dataset
    It's possible to pass an existing psycopg2 cursor connection, otherwise pass the parameters to open a new connection
    :param datasetid:
    :param colname: the boolean column to update
    :param value: True or False
    :param conndict: dictionary with postgresql connection, passs this with cur=None to make anew connection
            e.g. {"dbname": "grapes", "user": "claudio", "password": "claudio", "port":"5432", "host": "127.0.0.1"}
    :param cur: existing cursor, pass this if you want to use an existing connection
    :return:
    """

    hadcursor = True if cur else False #set if we are using an already existing connection

    try:

        if not cur:
            conn = pgconnect(**conndict)
            cur = conn.cursor()

            str1="FALSE"
            if value==True:
                str1="TRUE"

            cur.execute("""UPDATE colorgrade
                            SET %s="""+str1+"""
                               from (
                                        SELECT data.colorgrade.id_cgrade
                                        FROM data.SensoDatum
                                        INNER JOIN data.colorgrade
                                        ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                                        where data.SensoDatum.id_sensor= %s
                                    ) as x
                             WHERE colorgrade.id_cgrade = x.id_cgrade""", (AsIs(colname),datasetid))

    except Exception as e:
        print(e)
        raise (e)
    finally:
        if not hadcursor:  # if we passed the cursor we will not close the connection in this function
            if cur:
                cur.close()
            if conn:
                conn.close()

def filter_bystep(conndict, datasetid, step=2, boundid=None,  outpath="/vagrant/code/STARS/berry/data/", export=True, debug=True):
    """
    Reduce the number of sensor points using a step
    reduction is based on a specific colorgrade field.
    The user can choose to restrict the filter to a polygon boundary.
    the function set the 'keep' fields to True for those points that we want to keep,
    result can be saved as a csv on disk (recommended in a multiuser database!!!)

    note: the filtering is happening on a file on disk

    :param conndict: dictionary with postgresql connection
            e.g. {"dbname": "grapes", "user": "claudio", "password": "claudio", "port":"5432", "host": "127.0.0.1"}
    :param datasetid:
    :param step: reduction step (1 keep all values, 2 skip 1 value, 3 skip 2 values,...
    :param boundid: the id of the clipping polygon
    :param outpath:
    :param export:
    :param debug:
    :return:
    """

    #create unique name for temporary tables, also add a front letter and delete - to get a correct table name
    tblname = utils.create_uniqueid()
    tblname1 = utils.create_uniqueid()
    conn = ""
    cur = ""
    f=""
    tempfile= outpath + "/temp_"+tblname #define a unique name for a temporary file
    outpath = outpath + "/step"+str(step)
    try:

        conn = pgconnect(**conndict)
        cur = conn.cursor()

        #create text file in write mode
        f = open(tempfile, "w")

        #use COPY to download id from the colorgrade table , if there is a boundary clip the data
        str1 = """,data.SensoDatum.geom""" if boundid else """ """
        corequery1 = """SELECT data.colorgrade.id_cgrade"""+str1+"""
                        FROM data.SensoDatum
                        INNER JOIN data.colorgrade
                        ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                        where data.SensoDatum.id_sensor=""" + str(datasetid)
        if boundid:
            cur.copy_expert("""COPY (
                                select k.id_cgrade from ("""
                             + corequery1 +
                            """) as k
                             INNER JOIN
                             data.boundary
                             ON ST_Contains(boundary.geom, k.geom)
                             where boundary.id_bound = """+ str(boundid)+"""
                             order by k.id_cgrade ) TO STDOUT WITH CSV""", f)
        else:
            cur.copy_expert("""COPY ("""
                             + corequery1 +
                            """ order by id_cgrade ) TO STDOUT WITH CSV""", f)

        f.close()

        ###############TODO: can I do the filter by step inside the database
        #open with numpy,filter,save to disk
        data = np.loadtxt(tempfile, delimiter =",")
        nrows = data.shape[0]
        filtered = data[range(0,nrows,step)]
        np.savetxt(tempfile, filtered, fmt="%.f", delimiter=",")
        ##############

        # create temporary empty table with column id
        cur.execute("""CREATE TEMP TABLE %s(
                        ID   serial  primary key
                        ) ON COMMIT DROP""", (AsIs(tblname),))
        f = open(tempfile, "r")
        #upload with COPY to the temporary table
        cur.copy_expert("""COPY """+tblname+""" FROM STDIN WITH CSV""", f)


        corequery2 = """     SELECT data.SensoDatum.row, data.SensoDatum.geom, data.colorgrade.*
                            --FROM data.sensor
                            --INNER JOIN
                            FROM data.SensoDatum
                            --ON data.sensor.id_sensor = data.SensoDatum.id_sensor
                            INNER JOIN data.colorgrade
                            ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                            where data.SensoDatum.id_sensor= %s"""
        if boundid: #if we choose  polytgon boundary we need to select the points inside
            cur.execute("""     CREATE TEMP TABLE %s ON COMMIT DROP AS(
                                select foo.*
                                from ("""+corequery2+""") as foo
                                INNER JOIN data.boundary
                                ON ST_Contains(boundary.geom, foo.geom)
                                where boundary.id_bound = %s
                                )
                                """, (AsIs(tblname1), datasetid, boundid))
        else:
            cur.execute(""" CREATE TEMP TABLE %s ON COMMIT DROP AS(""" + corequery2 + """)""",
                        (AsIs(tblname1), datasetid))

        #initialize the field keep to FALSE for the current dataset
        cur.execute("""update data.colorgrade
                        set keepA = FALSE,keepB = FALSE,keepC = FALSE,keepD = FALSE
                        from (select id_cgrade from %s) as x
                        WHERE data.colorgrade.id_cgrade = x.id_cgrade""",(AsIs(tblname1),))

        '''
        #initialize the field keep to FALSE for the dataset
        cur.execute("""update data.colorgrade
                        set keepA = FALSE,keepB = FALSE,keepC = FALSE,keepD = FALSE
                        from (SELECT data.colorgrade.ID_cgrade as ID_cgrade
                              FROM data.sensor
                              INNER JOIN
                              data.SensoDatum
                              ON data.sensor.id_sensor = data.SensoDatum.id_sensor
                              INNER JOIN data.colorgrade
                              ON data.SensoDatum.id_sdata = data.colorgrade.id_sdata
                              where data.sensor.id_sensor= %s) as x
                        WHERE data.colorgrade.id_cgrade = x.id_cgrade""", (datasetid,))
        '''

        #use join between colorgrade and temporary table to update the keep field
        cur.execute("""update data.colorgrade
                        set keepA = TRUE, keepB = TRUE,keepC = TRUE,keepD = TRUE
                        from (select id from %s ) as x
                        WHERE data.colorgrade.id_cgrade = x.id""",(AsIs(tblname),))

        #TODO test if this exporting is working properly (multiaccess problems)
        if export:
            if boundid:
                #i am passing the cursor so that I can stay in the same transaction (hopefully!!!)
                download_colordata(datasetid, allcolors=True, outfile=outpath, boundid=boundid, filtered=True, cur=cur)
            else:
                download_colordata(datasetid, allcolors=True, outfile=outpath, filtered=True, cur=cur)

        #commit changes
        conn.commit()

        #vacuum colorgrade
        conn.autocommit = True
        cur.execute("""VACUUM ANALYZE data.ColorGrade""")

    except Exception as e:
        print(e)
        if conn:
            conn.rollback()
        raise(e)

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
        if f:
            f.close()

def visualize():
    raise NotImplementedError

def create_process(conndict, identifier, username, ptype):
    """
    Fuction to insert a new row in the processing table
    for the json structure refer to the database design documentation
    
    :param conndict: dictionary with postgresql connection
            e.g. {"dbname": "grapes", "user": "claudio", "password": "claudio", "port":"5432", "host": "127.0.0.1"}
    :param identifier: the operation unique identifier 
    :param username: username
    :param ptype: geoprocessing type
    :return:
    """

    conn=""
    cur=""
    try:
        #open a connection
        conn = pgconnect(**conndict)
        #insert table row if the user is not authorized
        cur = conn.cursor()
        cur.execute("""SELECT ID_user,authorized,active from data.Duser where name=%s""", (username,))
        auth = cur.fetchone()
        if not auth:
            conn.commit()
            raise Exception("User "+ username +" does not exist!")
        if not auth[2]:
            rst = json.dumps(
                {
                    "result":[],
                    "error": "User " + username + " is not active!"
                }
            )
            cur.execute("""INSERT INTO data.Processing (ID_user,uid, message,type, result) VALUES (%s,%s,%s,%s,%s)""",
                        (auth[0], identifier, geomessages.unauthorized.name, ptype,rst))
            conn.commit()
            raise PermissionError(rst)
        if auth[1] ==False:

            rst = json.dumps(
                {
                    "result":[],
                    "error": "User "+ username +" is not authorized to geoprocessing"
                }
            )
            cur.execute("""INSERT INTO data.Processing (ID_user,uid, message,type, result) VALUES (%s,%s,%s,%s,%s)""",
                        (auth[0], identifier, geomessages.unauthorized.name, ptype,rst))
            conn.commit()
            raise PermissionError(rst)

        else:
            cur.execute("""INSERT INTO data.Processing (ID_user,uid,message,type) VALUES (%s,%s,%s,%s)""",
                        (auth[0], identifier,geomessages.running.name, ptype))
            conn.commit()
    except Exception as e:
        if cur:cur.close()
        if conn: conn.rollback()
        raise e
    finally:
        if cur:cur.close()
        if conn: conn.close()


def update_process(conndict, identifier, username, message, result='{}'):
    """ 
    Update a process in the processing table
        
    :param conndict: dictionary with postgresql connection
            e.g. {"dbname": "grapes", "user": "claudio", "password": "claudio", "port":"5432", "host": "127.0.0.1"}
    :param identifier:  the operation unique identifier in UUID format
    :param username: username (this is used only to check if the user exists)
    :param message: message to update
    :param result: json message
    
    the structure on 4/4/17 is 
    {
        "result": [{
            "type": "",
            "path": ""
        }, {….}  ],
        "error": {
            "message": ""
        }
    }

    For the current json structure refer to the database design documentation    
        :return:
    """


    conn=""
    cur=""
    try:
        #open a connection
        conn = pgconnect(**conndict)
        #insert table row if the user is not authorized
        cur = conn.cursor()
        cur.execute("""SELECT ID_user from data.Duser where name=%s""", (username,))
        auth = cur.fetchone()
        if not auth: raise Exception("User " + username + " does not exist!")

        dt = datetime.datetime.now()
        cur.execute("""update data.processing set finish = %s, message = %s, result = %s where uid = %s AND ID_user=%s  """,
                    (dt, message, result, identifier,auth[0]))
        conn.commit()

    except Exception as e:
        if cur:cur.close()
        if conn: conn.rollback()
        raise e
    finally:
        if cur:cur.close()
        if conn: conn.close()


def create_processinfo(identifier, ptype, folder = "/temp"):
    """
    this function will create a json file with process info and put in a shared folder
    the operation has a unique identifier
    the client can query this file to get updates
    the message will be updated during the operation
    at successful completion the message will contain the url with the process result

    {
        id: uniqueid
        user:
        type: processtype (e.g. datafiltering)
        start: starttime
        end: endtime
        message: e.g "running","stopped","error","completed", "unauthorized"
        url: the url where a result is stored on disk
    }

    :param identifier: unique identifier for this operation (will be used as the file name)
    :param folder: folder where the json file will be stored
    :return:
    """

    '''
    https://github.com/esnme/ultrajson faster than json and simplejson
    '''

    raise NotImplementedError('using database table')


def update_processinfo(identifier, text, folder = "/temp"):
    """
    As the oparation proceeds the the json file is updated
    :param identifier: unique identifier for this operation (will be used as the file name)
    :param folder: folder where the json file will be stored
    :param text: dictionary with values to update
    :return:
    """

    # open json file
    # update file
    # save file

    ''' example with json module
    with open('my_file.json', 'r+') as f:
        json_data = json.load(f)
        json_data['b'] = "9"
        f.seek(0)
        f.write(json.dumps(json_data))
        f.truncate()
    '''
    raise NotImplementedError('using database table')


if __name__ == "__main__":


    #######TESTS#############

    connstring = {"dbname": "grapes", "user": "claudio", "password": "claudio", "port":"5432", "host": "127.0.0.1"}

    ###############test connection###########
    #conn= pgconnect(**connstring)
    #print(conn)
    #conn.close()


    ######cretaing/updating processing table##########
    #identifier= str(uuid.uuid4())
    #create_process(connstring, identifier, "claudio", "trial") #authorized user
    #create_process(connstring, identifier, "inactive", "trial") #inactive user
    #create_process(connstring, identifier, "unauthorized", "trial") #unauthorized user
    #create_process(connstring, identifier, "ghost", "trial") # non existing user

    #identifier="f2ebf25b-f707-49dd-ae6b-f573ad5c3a50"
    #js = json.dumps({"type": "database", "path": {"table": "ciccio", "id": 2}})
    #update_process(connstring, identifier, "claudio", geomessages.completed.name, result= js) #update with json
    #update_process(connstring, identifier, "claudio", geomessages.stopped.name ) #empty result and stopped
    #update_process(connstring, identifier, "claudio", geomessages.error.name, ) #error
    #update_process(connstring, identifier, "mrx", geomessages.error.name, ) # non existing user
    #

    ############ uploading csv
    datasetid=26
    boundid = 1
    #csvToPostgres("2016_06_17.csv", "","","","")

    #######statistics
    #get_statistics(connstring, 26, "a") #no rows, no boundary
    #get_statistics(connstring, 26, "a",True)  #with rows, no boundary
    #get_statistics(connstring, 26, "a", boundid=1)#no rows, with boundary
    #get_statistics(connstring, 26, "a",userows=True, boundid=1)  # with rows, with boundary

    ######download color data
    #all colors, no filter, no boundary
    #download_colordata(connstring, datasetid, colname="d", allcolors=True, boundid=None,filtered= False)
    #all colors, no filter, with boundary
    #download_colordata(connstring, datasetid, colname="d", allcolors=True, boundid=1,filtered= False)
    #all colors, with filter, no boundary
    #download_colordata(connstring, datasetid, colname="d", allcolors=True, boundid=None,filtered= True)
    #all colors, with filter, with boundary #this is slow!!! #TODO can i make this faster?
    #download_colordata(connstring, datasetid, colname="d", allcolors=True, boundid=1, filtered=True)
    # one column, no filter, no boundary
    #download_colordata(connstring, datasetid, colname="a", allcolors=False, boundid=None, filtered=False)
    # one column, no filter, with boundary
    #download_colordata(connstring, datasetid, colname="d", allcolors=False, boundid=1,filtered= False)
    # one column, with filter, no boundary
    #download_colordata(connstring, datasetid, colname="d", allcolors=False, boundid=None,filtered= True)
    # one column, with filter, with boundary
    #download_colordata(connstring, datasetid, colname="b", allcolors=False, boundid=1, filtered=True)

    #dowload passing a cursor
    '''
    conn="";cur=""
    try:
        conn= pgconnect(**connstring)
        print(conn)
        cur = conn.cursor()
        download_colordata(connstring, datasetid, colname="b", allcolors=False, boundid=1, filtered=True, cur=cur)
    finally:
        cur.close()
        conn.close()
    '''

    ############ filter by std
    #filter_bystd(connstring, datasetid , nstd=1, colname="a", userows=True, boundid=1)
    #filter_bystd(connstring, datasetid , nstd=1, colname="a", userows=False, boundid=1)
    #filter_bystd(connstring, datasetid , nstd=1, colname="a", userows=True)
    #filter_bystd(connstring, datasetid , nstd=1, colname="a", userows=False)

    ############ filter by value
    ###  keep >= value
    #delete 0, no boundary
    #filter_byvalue(connstring, datasetid, colname="a", value=0.8, filterzero=True, operator=">=", boundid=None)
    #delete 0, with boundary
    #filter_byvalue(connstring, datasetid, colname="a", value=0.8, filterzero=True, operator=">=", boundid=1)
    #keep 0, no boundary
    #filter_byvalue(connstring, datasetid, colname="a", value=0.8, filterzero=False, operator=">=", boundid=None)
    #keep 0, with boundary
    #filter_byvalue(connstring, datasetid, colname="a", value=0.8, filterzero=False, operator=">=", boundid=1)
    ##### keep <= #TODO the result keep values <= and not <
    # delete 0, no boundary
    #filter_byvalue(connstring, datasetid, colname="a", value=0.8, filterzero=True, operator="<=", boundid=None)
    # delete 0, with boundary
    #filter_byvalue(connstring, datasetid, colname="a", value=0.8, filterzero=True, operator="<=", boundid=1)
    # keep 0, no boundary
    #filter_byvalue(connstring, datasetid, colname="a", value=0.8, filterzero=False, operator="<=", boundid=None)
    # keep 0, with boundary
    #filter_byvalue(connstring, datasetid, colname="a", value=0.8, filterzero=False, operator="<=", boundid=1)


    ######filter by step
    #filter_bystep(connstring,datasetid,step=2, boundid=None)
    #filter_bystep(connstring,datasetid,step=2, boundid=1)


    ######### processing with logging
    '''
    try:
        identifier = str(uuid.uuid4()) #the identifier should be sent by the client
        #user = "inactive"
        #user = "unauthorized"
        user = "claudio"
        create_process(connstring, identifier, user, geotypes.statistics.name)
        get_statistics(connstring, 26, "a")
    except PermissionError as e:
        print(e)
    except Exception as e:
        #print(e)
        update_process(connstring, identifier, user, geomessages.error.name, result=json.dumps({"error": str(e)}))
    else:
        update_process(connstring, identifier, user, geomessages.completed.name, result= json.dumps({"message": "ok"}))

    '''




    #####################################################################################
    ########### INTERPOLATION ###########################################################


    #idea multiple filters: pass the connection to filters, do not commit inside a filter, set the output to file only
    #for the last filter,  (but there is the problem of passing the filtered dataset ( maybe just reaset the keep
    #of the first filter then  avoid to reset the others and add some selection of the true values in the sql)


    # 1 download filtered data with the chosen
    connstring = {"dbname": "grapes", "user": "claudio", "password": "claudio", "port": "5432", "host": "127.0.0.1"}
    outfile = "/vagrant/code/STARS/berry/interpolation/trial"
    #download_colordata(26, allcolors=True,conndict=connstring, outfile="/vagrant/code/STARS/berry/interpolation/trial")

    #filter_byvalue(connstring, 26, colname="D", value=0.8, filterzero=True, operator=">=", boundid=None,
                   #outpath="/vagrant/code/STARS/berry/interpolation/", export=True)
    from pysdss.filtering import filter

    #df = pd.read_csv("/vagrant/code/STARS/berry/interpolation/trial"+".csv")

    #df = filter_byvalue(df, value=0.0, operator=">", colname="harvestable_a", userows=False)



    #data = np.loadtxt(outfile+".csv", delimiter=",")
    #nrows = data.shape[0]
    #filtered = data[range(0, nrows, step)]


    #if necessary delete points where the interesting column is zero


    # define an interpolation grid (consider the user boundary)
    # log a running process to the database
    # save grid metadata and interpolation metadata to the database historic tables
    # execute interpolation
    # add result url to the database table
    # log completed process (maybe should add the location of the result








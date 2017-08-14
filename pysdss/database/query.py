# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        query.py
# Purpose:     query postgresql database to select/add/delete data
#
# Author:      claudio piccinini
#
# Updated:     10/08/2017
#-------------------------------------------------------------------------------

import psycopg2
from psycopg2.extensions import AsIs

from pysdss.settings import default_connection
from pysdss.database import pgconnect

def get_records(table, fields, conndict=None, cur=None):
    """
    return the id and a field
    :param table: table to query , can be YieldType, SoilType, CanoType, SensoType
    :param fields: list od fields to query
    :param conndict: dictionary with connection
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
    set a field
    :param table: table to query , can be YieldType, SoilType, CanoType, SensoType
    :param tpe: string with the type to add
    :param conndict: dictionary with connection
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
    Delete a record
    :param table:
    :param field:
    :param value:
    :param conndict:
    :param cur:
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


if __name__ == "__main__":

    def tests():
        ##tests####
        set_record('data.sensotype', "type", "remote sensing", conndict=default_connection)
        print(get_records('data.sensotype',("id_stype","type"), conndict=default_connection))
        delete_record('data.sensotype', "id_stype",18, conndict=default_connection)
    #tests()

# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        utils.py
# Purpose:     widely used functions
#
#               -   run an executle with python
#               -   run a python script
#               -   filter files by extension name and return of list of names
#               -   create a unique name for temporary database tables or filenames based on uuid
#               -   save load json files
#               -   Remove duplicate rows for the considered column values in a csv or pandas dataframe
#               -   Calculate the average distance between points
#               -   Calculate vineyard row length
#               -   function that returns a list of indices for a given simple value in a list l
#               -   shuffle a 2d numpy array and split in training and validation
#               -   get possible band combinations ( where band1-band2 is different from band2- band1)
#               -   Convert a list of tuples,each tuple with a band combination, to a list of formatted strings
#               -   Replace values in a dataframe
#               -   Calculate the root mean square error
#
# Author:      claudio piccinini
#
# Updated:     22/02/2017
#-------------------------------------------------------------------------------

import uuid
import json
import os
import statistics
import math

import pandas as pd
import numpy as np


def run_tool(params):
    """ run an executable tool (exe, bat,..)
    :param params: list of string parameters  ["tool path", "parameter1", "parameter2",.... ]
    :return: messages
    """
    import subprocess
    p = subprocess.Popen(params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # A pipe is a section of shared memory that processes use for communication.
    out, err = p.communicate()
    return bytes.decode(out), bytes.decode(err)


def run_script(params, callpy= ["py","-3.5"]):
    """ execute a python script
    :param params: a list of strings [ 'python version' , 'parameters']
    :param callpy: how to start the python interpreter
    :return: script message 
    """
    import subprocess
    #params.insert(0,callpy)
    params = callpy + params
    p = subprocess.Popen(params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #print(params)
    out, err= p.communicate()
    #return bytes.decode(out)+'\n'+ bytes.decode(err)
    return bytes.decode(out), bytes.decode(err)


def filter_files(basepath, filter):
    """ filter files by extension name and return of list of names
    :param path: tha path to the folder
    :param filter: a list with the file extensions   e.g. ['.tif']
    :return: a list of file names
    """
    # get the content of the images directory
    f = os.listdir(basepath)
    a = []
    # iterate all the files and keep only the ones with the correct extension
    for i in f:
        # filter content, we want files with the correct extension
        if os.path.isfile(basepath+'/'+i) and (os.path.splitext(basepath+'/'+i)[-1] in filter):
            a.append(i)
    return a


def create_uniqueid(start="a"):
    """
    create a unique name for temporary database tables or filenames
    required:import uuid
    :param start: starting character
    :return: unique id string
    """
    return "a" + str(uuid.uuid4()).replace("-", "")


def json_io(path, obj=None, direction="out", **kwargs):
    """
    Save load json
    :param path: the path to the file
    :param obj: object to save
    :param direction: "out" for saving; "in" for input
    :param kwargs: additional arguments to pass to the functions
    :return: object if input
    """
    if direction == "out":
        with open(path, "w") as f:
            f.write(json.dumps(obj, **kwargs))
    else:
        with open(path, "r") as f:
            return json.load(f, **kwargs)


def remove_duplicates(df, cols, operation=None):
    """
    Remove duplicate rows for the considered column values
    :param df: pandas dataframe
    :param cols: list of colum names
    :param operation: pass "mean", "sum", "tail" , by default the "head" is used 
    :return: the filtered dataframe
    """

    if type(df) == str:
        #df = pd.read_csv(df, usecols=cols)
        df = pd.read_csv(df)
    elif type(df) == pd.core.frame.DataFrame:
        pass
    else:
        raise TypeError('pass a csv or a pandas dataframe')

    #grouped = df.groupby(cols, as_index=False)
    #return grouped.apply(lambda x: x.mean(),sorted=False)

    if operation == "mean":
        c= df.columns #store the original column order
        return df.groupby(cols,as_index=False, sort=False).mean()[c] #apply average to grops and return the original column order

    #TODO how do i do the sum without changing the vineyard row number?
    elif operation == "sum":
        raise NotImplementedError("the sum will change the row number, therefore the parameter is not allowed")
        c= df.columns #store the original column order
        return df.groupby(cols,as_index=False, sort=False).sum()[c] #apply average to grops and return the original column order

    elif operation == "tail":
        c= df.columns #store the original column order
        return df.groupby(cols,as_index=False, sort=False).tail(1)[c] #apply average to grops and return the original column order

    else:  #return the first item of the group
       c= df.columns #store the original column order
       return df.groupby(cols,as_index=False, sort=False).head(1)[c] #apply average to grops and return the original column order'''
    '''
    else:
        grouped = df.groupby(cols)
        index = [gp_keys[0] for gp_keys in grouped.groups.values()]
        index.sort() #sort the indexes otherwise the output will lose the original ordering
        unique_df = df.reindex(index)
        return unique_df'''


'''
def duplicated(list values, take_last=False):
    """
    cython implementation of remove_duplicates based on
    http://wesmckinney.com/blog/filtering-out-duplicate-dataframe-rows/
    """

    cdef:
        Py_ssize_t i, n
        dict seen = {}
        object row

    n = len(values)
    cdef ndarray[uint8_t] result = np.zeros(n, dtype=np.uint8)

    if take_last:
        for i from n > i >= 0:
            row = values[i]
            if row in seen:
                result[i] = 1
            else:
                seen[row] = None
                result[i] = 0
    else:
        for i from 0 <= i < n:
            row = values[i]
            if row in seen:
                result[i] = 1
            else:
                seen[row] = None
                result[i] = 0

    return result.view(np.bool_)
'''


def average_point_distance(file, xfield, yfield, row, direction="x", rem_duplicates = False, operation="mean"):
    """
    Calculate the average distance between points 
    The file must have the vineyard rows
    
    :param file: path to csv or pandas dataframe
    It's up to the user to pass a file with the three important fields (x,y,row)
    :param xfield: field name for x coordinates
    :param yfield: field name for y coordinates
    :param row: field name for rows
    :param direction: direction of the vineyard rows ( "x" other values are considered "y" )
    :param delete_duplicates: delete duplicate points
    :return:  the average distance
    """

    #1 open the csv file to pandas
    if type(file) == str:
        file = pd.read_csv(file, usecols=[xfield,yfield,row])
    elif type(file) == pd.core.frame.DataFrame:
        pass
    else:
        raise TypeError('pass a csv or a pandas dataframe')

    #2 delete duplicate points to avoid distances equals to zero
    if rem_duplicates:
        file = remove_duplicates(file, [yfield, xfield],operation=operation)

    #3 initialize an empty array with 4 columns (row, coordinates, shifted coordinates, distance)
    arr = np.zeros((file.shape[0],4))

    # first column with the rows
    arr[:, 0] = file[row].values
    #second column with the x/y coordinates
    if direction == "x":  # extract the x coordinates
        arr[:, 1] = file[xfield].values
    else:  # extract the y coordinates
        arr[:, 1] = file[yfield].values

    #third column with the shifted coordinates
    #copy the shifted by -1 coordinates to the third column
    arr[:-1, 2] = arr[1:,1]

    #fourth column with the differences
    #calculate the absolute difference of the coordinates and put in a column
    arr[:, 3] = np.abs(arr[:, 1] - arr[:, 2])

    # clacualte averages for each row and then final average
    means = []
    rows = np.unique(arr[:, 0]) #get unique rows
    for x in rows:
        r = np.where(arr[:, 0] == x) #get the indexes for this row
        # skip the last point becuse contains distances between point on different rows, coloumn 4 has the average values
        means.append(np.mean(arr[r][:-1,3]))
    # finally return an average for all the rows
    # print(means)
    return statistics.mean(means)


def max_point_distance(file, xfield, yfield, row, direction="x",  rem_duplicates = False, operation="mean"):
    """
    Calculate the max distance between points
    The file must have the vineyard rows

    :param file: path to csv or pandas dataframe
    It's up to the user to pass a file with the three important fields (x,y,row)
    :param xfield: field name for x coordinates
    :param yfield: field name for y coordinates
    :param row: field name for rows
    :param direction: direction of the vineyard rows ( "x" other values are considered "y" )
    :param delete_duplicates: delete duplicate points
    :return:  the max distance
    """

    # 1 open the csv file to pandas
    if type(file) == str:
        file = pd.read_csv(file, usecols=[xfield, yfield, row])
    elif type(file) == pd.core.frame.DataFrame:
        pass
    else:
        raise TypeError('pass a csv or a pandas dataframe')

    # 2 delete duplicate points to avoid distances equals to zero
    if rem_duplicates:
        file = remove_duplicates(file, [yfield, xfield], operation=operation)

    # 3 initialize an empty array with 4 columns (row, coordinates, shifted coordinates, distance)
    arr = np.zeros((file.shape[0], 4))

    # first column with the rows
    arr[:, 0] = file[row].values
    # second column with the x/y coordinates
    if direction == "x":  # extract the x coordinates
        arr[:, 1] = file[xfield].values
    else:  # extract the y coordinates
        arr[:, 1] = file[yfield].values

    # third column with the shifted coordinates
    # copy the shifted by -1 coordinates to the third column
    arr[:-1, 2] = arr[1:, 1]

    # fourth column with the differences
    # calculate the absolute difference of the coordinates and put in a column
    arr[:, 3] = np.abs(arr[:, 1] - arr[:, 2])

    # clacualte max for each row and then highest max
    maxim = []
    rows = np.unique(arr[:, 0])  # get unique rows
    for x in rows:
        r = np.where(arr[:, 0] == x)  # get the indexes for this row
        # skip the last point becuse contains distances between point on different rows, coloumn 4 has the average values
        maxim.append(np.max(arr[r][:-1, 3]))
    # finally return the highest for all the rows
    return max(maxim)


def max_point_distance_byrow(pdf, xfield, yfield, direction="x", rem_duplicates = False, operation="mean"):
    """
    Calculate the max distance between points
    The file must have the vineyard rows

    :param pdf: pandas dataframe
    It's up to the user to pass a dataframe with the 2 fields (x,y)
    :param xfield: field name for x coordinates
    :param yfield: field name for y coordinates
    :param direction: direction of the vineyard rows ( "x" other values are considered "y" )
    :param delete_duplicates: delete duplicate points
    :return:  the max distance
    """

    # 1 open the csv file to pandas
    if type(pdf) != pd.core.frame.DataFrame:
        raise TypeError('pass a csv or a pandas dataframe')

    # 2 delete duplicate points to avoid distances equals to zero
    if remove_duplicates:
        pdf = rem_duplicates(pdf, [yfield, xfield], operation=operation)

    # 3 initialize an empty array with 3 columns (coordinates, shifted coordinates, distance)
    arr = np.zeros((pdf.shape[0], 3))

    # first column with the x/y coordinates
    if direction == "x":  # extract the x coordinates
        arr[:, 0] = pdf[xfield].values
    else:  # extract the y coordinates
        arr[:, 0] = pdf[yfield].values

    # second column with the shifted coordinates
    # copy the shifted by -1 coordinates to the second column
    arr[:-1, 1] = arr[1:, 0]

    # third column with the differences
    # calculate the absolute difference of the coordinates and put in a column
    arr[:, 2] = np.abs(arr[:, 0] - arr[:, 1])

    # return max distance for the current row, skip the last element which is zero
    return np.max(arr[:-1, 2])


def row_length(file, xfield, yfield, row, conversion_factor=1):
    """
    Return the lenght of the vineyard rows as an 2 columns array row_id - length
    The file must have the vineyard rows and projected coordinates
    
    :param file: path to csv or pandas dataframe
    It's up to the user to pass a file with the three important fields (x,y,row)
    :param xfield: field name for x coordinates
    :param yfield: field name for y coordinates
    :param row: field name for rows
    :param conversion_factor: this is used if to change the lenght units 
           e.g to convert from meters to feet use  3.2808398950131 ft
                  to convert from feet to meters use 0.3048 m
    :return:  an array with columns  row_id - length
    """

    # open the csv file to pandas
    if type(file) == str:
        file = pd.read_csv(file, usecols=[xfield,yfield,row])
    elif type(file) == pd.core.frame.DataFrame:
        pass
    else:
        raise TypeError('pass a csv or a pandas dataframe')

    # get unique row names and sort them
    rows = file[row].unique()
    rows.sort()

    # initialize the output empty array
    arr = np.zeros((len(rows), 2))

    # for each row name
    for i, r in enumerate(rows):

        # select one vineyard row
        fdf = file[file[row] == r]

        # get first last point coordinates
        x = fdf.head(1)[xfield].values[0] - fdf.tail(1)[xfield].values[0]
        y = fdf.head(1)[yfield].values[0] - fdf.tail(1)[yfield].values[0]

        # add length to array
        arr[i, :] = r, math.hypot(x, y) * conversion_factor

    #return arr[arr[:, 0].argsort()] #not necessary as we already sorted the rows before iteration
    return arr


def get_indexes(l,value):
    """ function that returns a list of indices for a given simple value in a list l
    :param l: a list
    :param value: a simple value (character or number)
    :return: the list of indices
    """

    m = l.copy()  # create a copy of the list
    idx = []
    try:
        while True:
            ind = m.index(value)
            idx.append(ind)
            m[ind] = value/2  # create a fake value to continue the iteration
    except:  # this happens when the value is not found
            return idx


def shuffle_data(data, percentage):
    """ shuffle a 2d numpy array and split in training and validation

    :param data: 2d numpy array with all the data (samples, labels)
    :param percentage: percentage of validation data(integer)
    :return: trainingSamples,trainingLabels,validationSamples,validationLabels, number of training data
    """

    count = data.shape[0]  #nuber of rows

    # shuffle all the data randomly
    np.random.shuffle(data)

    # get number of training data
    k = int(math.ceil(count*((100-percentage)/100.0)))

    # set training data

    trainingSamples = data[:k, :-1]
    trainingLabels = data[:k, -1:]

    # set validation data
    validationSamples= data[k:, :-1]
    validationLabels = data[k:, -1:]

    return trainingSamples,trainingLabels,validationSamples,validationLabels, k


def combination_count(nbands = 8):
    """ get possible band combinations ( where band1-band2 is different from band2- band1)
    :param nbands: number of bands
    :return: a tuple: the total number of band combinations, a list of tuples where each tuple is the band combination
    """

    n = nbands  # number of bands (get this from the numpy array)
    i = 1
    # sum = 0
    k = n + 1
    names1 = []
    names2 = []
    while i < n:

        # sum += (n - i)

        # left-right band combination names
        j = i + 1
        while j <= n:
            # names1.append(str(i)+"-"+str(j)+"\t")
            names1.append((i, j))
            j += 1

        # right - left band combination names
        j = n - i
        k -= 1
        while j >= 1:
            # names2.append(str(k) +"-"+ str(j) + "\t")
            names2.append((k, j))
            j -= 1
        i += 1

    # sum *=2
    names = names1+names2

    # return sum, names1+names2
    return len(names), names


def column_names_to_string(t,sep1="-", sep2="\t"):
    """ Convert a list of tuples,each tuple with a band combination, to a list of formatted strings
    :param t: a list of tuples, each tuple with a combination e.g [(1,2),(1,3),...]
    :param sep1: the separator within a couple of names
    :param sep2: the separator between names
    :return: a list of strings
    """
    text = []
    for i in t:
        text.append(str(i[0]) + sep1 + str(i[1]) + sep2)
    return text


def clean_dataframe (df,invalues,outvalues): # TODO check the use of  DataFrame.replace
    """ 
    Change the invalues with the outvalues in a dataframe
    :param df: the dataframe to clean
    :param inp: a list with the input values
    :param out: a list with the output values
    :return: 
    """
    if len(invalues) != len(outvalues): raise ValueError("the input and output lists should be the same length")
    for i,v in enumerate(invalues):
        msk = (df == v)
        df[msk] = outvalues[i]


def rmse(predictions, targets):
    """
    Calculate the root mean square error
    :param predictions: 1d numpy array
    :param targets: 1d numpy array
    :return: the rmse
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


if __name__ == "__main__":


    '''
    algo={
                "name": "average",
                "radius1":"1",
                "radius2":"1",
                "angle":"0",
                "minpoints":"1",
                "nodata":"0"
        }

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
        "-a": algo,
        "-q":"",
        "src_datasource":"",
        "dst_filename":""
    }

    path = os.path.dirname(os.path.abspath(__file__))+"/experiments/interpolation/test.json"
    print(path)
    json_io(path, obj=params, direction="out")
    print("done")
    print(json_io(path, direction="in"))
    '''


    def test_average_point_distance():
        folder = "/vagrant/code/pysdss/data/output/text/oldoutputs/afd3fcd91565847638e73034217121ff7/"
        #file = folder +  "/shape/points_update.csv"
        #avdist = average_point_distance(file, "x", "y", "row", direction="x")

        file = folder + "/afd3fcd91565847638e73034217121ff7_keep.csv"
        avdist = average_point_distance(file, "lon", "lat", "row", direction="x")

        print(avdist)

    #test_average_point_distance()

    def test_row_length():
        folder = "/vagrant/code/pysdss/data/output/text/oldoutputs/afd3fcd91565847638e73034217121ff7/"
        file = folder + "/afd3fcd91565847638e73034217121ff7_keep.csv"
        #using csv
        l = row_length(file,"lon", "lat", "row", conversion_factor=1)
        print(l)
        #using dataframe
        file = pd.read_csv(file, usecols=["lon", "lat", "row",])
        l1 = row_length(file, "lon", "lat", "row", conversion_factor=1)
        print(l1)

        #check the arrays are equals(shape and elements)
        print(np.array_equal(l, l1))

    #test_row_length()


    def test_remove_duplicates():

        import shutil
        folder = "/vagrant/code/pysdss/data/output/text/"
        id = create_uniqueid()
        if not os.path.exists(folder + id):
            os.mkdir(folder + id)
        outfolder = folder + id + "/"

        file = "/vagrant/code/pysdss/data/input/2016_06_17_samplemedium.csv"
        file = "/vagrant/code/pysdss/data/input/2016_06_17.csv"
        file = shutil.copy(file, outfolder + "/" + id + "_keep.csv")  # copy to the directory

        # 5 set data properties: correct field names if necessary
        usecols = ["%Dataset_id", " Row", " Raw_fruit_count", " Visible_fruit_per_m", " Latitude", " Longitude",
                   "Harvestable_A", "Harvestable_B", "Harvestable_C", "Harvestable_D"]
        new_column_names = ["id", "row", "raw_fruit_count", "visible_fruit_per_m", "lat", "lon", "a", "b", "c", "d"]

        # 6 set data properties: automatically calculate the average distance between points after checking  if
        # duplicate points exists, (the user can also set an average distance from the web application)

        df = remove_duplicates(file, [" Latitude", " Longitude"], operation="mean")
        #df = remove_duplicates(file, [" Longitude"," Latitude" ], operation="mean")
        df.to_csv(file+"_mean.csv", index=False)
        #TODO why sum sums the row numbers while average do not average the number???
        #df = remove_duplicates(file, [" Latitude", " Longitude"], operation="sum")
        #df.to_csv(file+"_sum.csv", index=False)
        df = remove_duplicates(file, [" Latitude", " Longitude"], operation="tail")
        df.to_csv(file+"_tail.csv", index=False)
        df = remove_duplicates(file, [" Latitude", " Longitude"])
        df.to_csv(file+"_head.csv", index=False)

    test_remove_duplicates()
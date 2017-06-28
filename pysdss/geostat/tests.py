# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        test.py
# Purpose:     testing geostatistics code
#
#
# Author:      claudio piccinini, based on the code book GIS Algorithms by Ningchuan Xiao
#
# Updated:     27/05/2017
#-------------------------------------------------------------------------------

from math import sqrt
import numpy as np
import scipy.spatial.ckdtree as kdtree

def read_data(fname):
    """
    Reads in data from a file. Each line in the file must have
    three columns: X, Y, and Value.
    Input
      fname: name of and path to the file
    Output
      x3: list of lists with a dimension of 3 x n
          Each inner list has 3 elements: X, Y, and Value
    """
    with open(fname, 'r') as f:
        x3 = [[float(x[0]),float(x[1]),float(x[2])] for x in [x.strip().split() for x in f.readlines()]]

        return np.array(x3)



#TODO here we get the nearest 10 points with iteration, we should use a spatial index approach instead
def prepare_interpolation_data(point, data, n=10):
    """Get the nearest points to a point
    :param point: a point object
    :param data: a nested list [[x,y,value],....]
    :param n: number of nearest points to the point x
    :return: the n nearest points to point and the average value of data as a list
    """
    vals = [z[2] for z in data]
    mean = sum(vals)/len(vals)
    dist = [sqrt((z[0]-point.x)**2 + (z[1]-point.y)**2) for z in data] #here use scipy pdist instead?
    outdata = [(data[i][0], data[i][1], data[i][2], dist[i]) for i in range(len(dist))]
    outdata.sort(key=lambda x: x[3])
    outdata = outdata[:n]
    return outdata, mean


def prepare_interpolation_data_array(point, data, n=10):
    """Get the nearest points to a point
    :param point: a point object
    :param data: a 2darray [[x,y,value],....]
    :param n: number of nearest points to the point x
    :return: the n nearest points to point and the average value of data as a list
    """
    # vals = [z[2] for z in data]
    # mean = sum(vals)/len(vals)

    mean = np.mean(data[:, 2])
    dist = np.sqrt(np.power(data[:, 0] - point.x, 2) + np.power(data[:, 1] - point.y, 2))

    outdata = np.hstack((data, np.expand_dims(dist, axis=0).T)) #dist was 1D
    srtd = outdata[outdata[:, 3].argsort()][:n,:]
    return srtd, mean


def prepare_interpolation_data_kdtree(point, data, n=10, tree=None):
    """Get the nearest points to a point
    :param point: point object
    :param data:  a 2d array [[x,y,value],....]
    :param n: the number of nearest points to search
    :param kdtree: kdtree object made with kdtree.cKDTree()
    :return: the n nearest points to point and the average
    """

    mean = np.mean(data[:,2])
    #we need the x,y coordinates and a point array
    point = np.array([point.x, point.y])

    if tree:
        distance, index = tree.query(point,n)
    else:
        xydata  = data[:,:2]
        #use kdtree to return the n nearest points
        distance, index = kdtree.cKDTree(xydata).query(point,n)

    #extract the points from the original data and add column with distance
    #data = data[index]
    return np.hstack((data[index],np.expand_dims(distance, axis=0).T)), mean




def rmse(empirical, theoretical):
    """ Calculate the root mean square error between empirical and theoretical semivariances
    :param empirical: 1-d array
    :param theoretical: 1-d array
    :return: root mean square error
    """

    return sqrt((np.sum(np.power(empirical - theoretical, 2)) / empirical.shape[0]))

if __name__ == "__main__":

    ###test semivariance,covariance anf fitting models

    def test_variancs_fitting():
        import os.path
        import numpy as np
        from bokeh.plotting import figure
        from bokeh.io import output_file,  save
        from bokeh.models import Label
        import variance as vrc
        import fit

        data = read_data(os.path.dirname(os.path.realpath(__file__)) + "/test_data/necoldem.dat")

        hh = 50 #half of the bin size
        lags = np.arange(0, 3000, hh)
        gamma = vrc.semivar(data, lags, hh)
        cov = vrc.covar(data,lags,hh)
        #print(gamma)
        #print(cov)

        #chart empirical semivariance and covariance
        output_file(os.path.dirname(os.path.realpath(__file__)) + "/test_data/necoldem.dat.html")
        f = figure()
        f.line(gamma[0, :], gamma[1, :], line_color="green", line_width=2, legend="Semivariance")
        f.square(gamma[0, :], gamma[1, :], fill_color=None, line_color="green", legend="Semivariance")
        f.line(cov[0, :], cov[1,:], line_color="red", line_width=2, legend="Covariance")
        f.square(cov[0,:], cov[1,:], fill_color=None, line_color="red", legend="Covariance")
        f.legend.location = "top_left"
        save(f)

        #fit models to find best ranges
        svs = fit.fitsemivariogram(data, gamma, fit.spherical)
        svl = fit.fitsemivariogram(data, gamma, fit.linear)
        svg = fit.fitsemivariogram(data, gamma, fit.gaussian)
        sve = fit.fitsemivariogram(data, gamma, fit.exponential)

        #chart the fitted models
        output_file(os.path.dirname(os.path.realpath(__file__)) + "/test_data/necoldem.dat.fitted.html")
        f = figure()

        f.circle(gamma[0], gamma[1], fill_color=None, line_color="blue", legend="Empirical")
        f.line(gamma[0], svs(gamma[0]), line_color="red", line_width=2, legend="Spherical")
        f.line(gamma[0], svl(gamma[0]), line_color="orange", line_width=2, legend="Linear")
        f.line(gamma[0], svg(gamma[0]), line_color="black", line_width=2, legend="Gaussiab")
        f.line(gamma[0], sve(gamma[0]), line_color="green", line_width=2, legend="Exponential")

        #calculate rmse for the models
        rmses = rmse(gamma[0], svs(gamma[0]))
        rmsel = rmse(gamma[0], svl(gamma[0]))
        rmseg = rmse(gamma[0], svg(gamma[0]))
        rmsee = rmse(gamma[0], sve(gamma[0]))

        # making multiline labels /n and <br/> do not work
        rmse_label = Label(x=150, y=110, x_units='screen', y_units='screen',
                         text="RMSE",render_mode='css',
                         background_fill_color='white', background_fill_alpha=1.0)
        rmse_label2 = Label(x=150, y=90, x_units='screen', y_units='screen',
                         text="Spherical: "+str(rmses),render_mode='css',
                         background_fill_color='white', background_fill_alpha=1.0)
        rmse_label3 = Label(x=150, y=70, x_units='screen', y_units='screen',
                         text="Linear:" +str(rmsel),render_mode='css',
                         background_fill_color='white', background_fill_alpha=1.0)
        rmse_label4 = Label(x=150, y=50, x_units='screen', y_units='screen',
                         text="Gaussian:"+str(rmseg),render_mode='css',
                         background_fill_color='white', background_fill_alpha=1.0)
        rmse_label5 = Label(x=150, y=30, x_units='screen', y_units='screen',
                         text="Exponential:"+str(rmsee),render_mode='css',
                         background_fill_color='white', background_fill_alpha=1.0)
        f.add_layout(rmse_label)
        f.add_layout(rmse_label2)
        f.add_layout(rmse_label3)
        f.add_layout(rmse_label4)
        f.add_layout(rmse_label5)


        f.legend.location = "top_left"
        save(f)



    def test_variance_fitting2():
        import os.path
        import numpy as np
        from bokeh.plotting import figure
        from bokeh.io import output_file,  save
        from bokeh.models import Label
        import variance as vrc
        import fit
        import time

        data = read_data(os.path.dirname(os.path.realpath(__file__)) + "/test_data/necoldem.dat")

        hh = 5 #half of the bin size
        lags = np.arange(0, 3000, hh)

        start = time.time()
        gamma = vrc.semivar(data, lags, hh)
        stop = time.time()
        duration = stop - start
        print(duration * 1000)

        print(gamma)

        lags = np.arange(0-hh, 3000-hh, 2*hh)

        start = time.time()
        gamma = vrc.semivar2(data, lags, hh)
        cov = vrc.covar(data,lags,hh)
        stop = time.time()
        duration = stop - start
        print(duration * 1000)

        print(gamma)



        #return

        #chart empirical semivariance and covariance
        output_file(os.path.dirname(os.path.realpath(__file__)) + "/test_data/necoldem.dat.html")
        f = figure()
        f.line(gamma[0, :], gamma[1, :], line_color="green", line_width=2, legend="Semivariance")
        f.square(gamma[0, :], gamma[1, :], fill_color=None, line_color="green", legend="Semivariance")
        f.line(cov[0, :], cov[1,:], line_color="red", line_width=2, legend="Covariance")
        f.square(cov[0,:], cov[1,:], fill_color=None, line_color="red", legend="Covariance")
        f.legend.location = "top_left"
        save(f)

        #fit models to find best ranges
        svs = fit.fitsemivariogram(data, gamma, fit.spherical,forcenugget=True)
        svl = fit.fitsemivariogram(data, gamma, fit.linear,forcenugget=True)
        svg = fit.fitsemivariogram(data, gamma, fit.gaussian,forcenugget=True)
        sve = fit.fitsemivariogram(data, gamma, fit.exponential,forcenugget=True)

        #chart the fitted models
        output_file(os.path.dirname(os.path.realpath(__file__)) + "/test_data/necoldem.dat.fitted.html")
        f = figure()

        f.circle(gamma[0], gamma[1], fill_color=None, line_color="blue", legend="Empirical")
        f.line(gamma[0], svs(gamma[0]), line_color="red", line_width=2, legend="Spherical")
        f.line(gamma[0], svl(gamma[0]), line_color="orange", line_width=2, legend="Linear")
        f.line(gamma[0], svg(gamma[0]), line_color="black", line_width=2, legend="Gaussiab")
        f.line(gamma[0], sve(gamma[0]), line_color="green", line_width=2, legend="Exponential")

        #calculate rmse for the models
        rmses = rmse(gamma[0], svs(gamma[0]))
        rmsel = rmse(gamma[0], svl(gamma[0]))
        rmseg = rmse(gamma[0], svg(gamma[0]))
        rmsee = rmse(gamma[0], sve(gamma[0]))

        # making multiline labels /n and <br/> do not work
        rmse_label = Label(x=150, y=110, x_units='screen', y_units='screen',
                         text="RMSE",render_mode='css',
                         background_fill_color='white', background_fill_alpha=1.0)
        rmse_label2 = Label(x=150, y=90, x_units='screen', y_units='screen',
                         text="Spherical: "+str(rmses),render_mode='css',
                         background_fill_color='white', background_fill_alpha=1.0)
        rmse_label3 = Label(x=150, y=70, x_units='screen', y_units='screen',
                         text="Linear:" +str(rmsel),render_mode='css',
                         background_fill_color='white', background_fill_alpha=1.0)
        rmse_label4 = Label(x=150, y=50, x_units='screen', y_units='screen',
                         text="Gaussian:"+str(rmseg),render_mode='css',
                         background_fill_color='white', background_fill_alpha=1.0)
        rmse_label5 = Label(x=150, y=30, x_units='screen', y_units='screen',
                         text="Exponential:"+str(rmsee),render_mode='css',
                         background_fill_color='white', background_fill_alpha=1.0)
        f.add_layout(rmse_label)
        f.add_layout(rmse_label2)
        f.add_layout(rmse_label3)
        f.add_layout(rmse_label4)
        f.add_layout(rmse_label5)


        f.legend.location = "top_left"
        save(f)


    test_variance_fitting2()





    # testing ordinary kriging for 2 points
    def test_ordinaryk():
        import os.path
        import numpy as np
        import variance as vrc
        import fit
        import test_data.point as p
        import kriging as kr

        data = read_data(os.path.dirname(os.path.realpath(__file__))  + "/test_data/necoldem.dat")

        print("calculte empirical semivariance")
        hh = 50
        lags = range(0, 3000, hh)
        gamma = vrc.semivar(data, lags, hh)
        #covariance = vrc.covar(data, lags, hh)

        print("fit semivariogram")
        # choose the model with lowest rmse
        semivariograms=[]
        semivariograms.append(fit.fitsemivariogram(data, gamma, fit.spherical))
        semivariograms.append(fit.fitsemivariogram(data, gamma, fit.linear))
        semivariograms.append(fit.fitsemivariogram(data, gamma, fit.gaussian))
        semivariograms.append(fit.fitsemivariogram(data, gamma, fit.exponential))
        rsmes = [rmse(gamma[0], i(gamma[0])) for i in semivariograms]
        semivariogram = semivariograms[rsmes.index(min(rsmes))]

        print("################test point 337000, 4440911 ")
        x = p.Point(337000, 4440911)
        p1 = prepare_interpolation_data(x, data)
        print("ordinary kriging")
        print(kr.ordinary(np.array(p1[0]), semivariogram)[0:2])
        print("simple kriging")
        print(kr.simple(np.array(p1[0]), p1[1], semivariogram)[0:2])

        print("################test a point from the original data")
        x = p.Point(data[1, 0], data[1, 1])
        p1 = prepare_interpolation_data(x, data)
        print("ordinary kriging")
        print(kr.ordinary(np.array(p1[0]), semivariogram)[0:2])
        print("simple kriging")
        print(kr.simple(np.array(p1[0]), p1[1], semivariogram)[0:2])

    #test_ordinaryk()


    def interpolate_surface():

        import os.path
        import numpy as np
        import variance as vrc
        import fit
        import test_data.point as p
        import kriging as kr

        from bokeh.plotting import figure
        from bokeh.io import output_file, save

        data = read_data(os.path.dirname(os.path.realpath(__file__))  + "/test_data/necoldem.dat")

        ##TODO the semivariogram should be computed for every surface cell!!!

        print("calculte empirical semivariance")
        hh = 50
        lags = range(0, 3000, hh)
        gamma = vrc.semivar(data, lags, hh)
        #covariance = vrc.covar(data, lags, hh)

        print("fit semivariogram")
        # choose the model with lowest rmse
        semivariograms=[]
        semivariograms.append(fit.fitsemivariogram(data, gamma, fit.spherical))
        semivariograms.append(fit.fitsemivariogram(data, gamma, fit.linear))
        semivariograms.append(fit.fitsemivariogram(data, gamma, fit.gaussian))
        semivariograms.append(fit.fitsemivariogram(data, gamma, fit.exponential))
        rsmes = [rmse(gamma[0], i(gamma[0])) for i in semivariograms]
        semivariogram = semivariograms[rsmes.index(min(rsmes))]


        # define surface properties
        x0,x1 = data[:,0].min(),data[:,0].max()
        y0,y1 = data[:,1].min(),data[:,1].max()

        dx,dy = x1-x0, y1-y0

        dsize =min(dx/100.0, dy/100.0)

        nx = int(np.ceil(dx/dsize))
        ny = int(np.ceil(dy/dsize))


        # initialize empty arrays to store interpolated surfaces and errors

        surface_ordinary = np.zeros((nx,ny))
        error_ordinary = np.zeros((nx,ny))
        surface_simple = np.zeros((nx,ny))
        error_simple = np.zeros((nx,ny))



        print("Executing simple and ordinary kriging, this will take some time")
        # execute kriging for each surface cell
        for i in range(nx):
            for j in range(ny):

                x = p.Point(x0+i*dsize, y0+j*dsize)
                new_data, mu = prepare_interpolation_data_array(x,data,n=10)    #### TODO: this should use spatial indexing

                ordinary = kr.ordinary(np.array(new_data), semivariogram)
                surface_ordinary[i,j] = ordinary[0]
                error_ordinary[i,j] = ordinary[1]

                simple = kr.simple(np.array(new_data), mu, semivariogram)
                surface_simple[i,j] = simple[0]
                error_simple[i,j] = simple[1]


        # save arrays to disk, array are transposed to have the correct x,y coordimates
        np.savez(os.path.dirname(os.path.realpath(__file__))  +"/test_data/surfaces.npz",surface_ordinary=surface_ordinary.T,
                 error_ordinary=error_ordinary.T, surface_simple=surface_simple.T, error_simple=error_simple.T,
                 surface_difference= surface_ordinary-surface_simple)


    def plot_surfaces():
        import os.path
        import numpy as np
        from bokeh.plotting import figure
        from bokeh.io import output_file, save
        from bokeh.models import LinearColorMapper, ColorBar
        import pandas as pd

        print("plotting surfaces")
        outs = ["surface_ordinary","error_ordinary","surface_simple","error_simple", "surface_difference"]
        surfaces = np.load(os.path.dirname(os.path.realpath(__file__))  +"/test_data/surfaces.npz")
        surface_ordinary = surfaces["surface_ordinary"]
        error_ordinary = surfaces["error_ordinary"]
        surface_simple = surfaces["surface_simple"]
        error_simple = surfaces["error_simple"]
        surface_difference = surfaces["surface_difference"]

        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__))  + "/test_data/necoldem.dat", header=None, sep=" ")

        xmin = df.values[:,0].min()
        ymin = df.values[:,1].min()

        dx = df.values[:, 0].max() - df.values[:, 0].min()
        dy = df.values[:, 1].max() - df.values[:, 1].min()

        for i in outs:
            output_file(os.path.dirname(os.path.realpath(__file__))  + "/test_data/necoldem_"+ i+ ".html", title="necoldem_"+ i)
            f = figure()

            color_mapper = LinearColorMapper(palette="Viridis256", low=eval(i).min(), high=eval(i).max())

            f.image(image=[eval(i)], x=xmin, y=ymin, dw=dx, dh=dy, palette="Viridis256")
            f.circle(df.values[:,0], df.values[:,1], fill_color=None, line_color="blue")
            color_bar = ColorBar(color_mapper=color_mapper , location=(0, 0))
            f.add_layout(color_bar, 'left')
            save(f)


    #interpolate_surface()
    #plot_surfaces()

    def test_distances_kdtree():
        import test_data.point as p
        import os.path
        import time

        data = read_data( os.path.dirname(os.path.realpath(__file__))  +"/test_data/necoldem.dat")
        point = p.Point(337000, 4440911)

        srtd, mean = prepare_interpolation_data_array(point, data, n=10)
        print(srtd)
        print(mean)

        srtd1, mean1 = prepare_interpolation_data_kdtree(point, data, n=10)

        print(srtd1)
        print(mean1)

        print(np.array_equal(srtd, srtd1))
        print(mean == mean1)


        print("times with array")
        for n in [10,50,100]:
            start = time.time()
            prepare_interpolation_data_array(point, data, n=n)
            stop = time.time()
            duration = stop - start
            print(duration*1000)

        print("times with kdtree")
        for n in [10,50,100]:
            start = time.time()
            prepare_interpolation_data_kdtree(point, data, n=n)
            stop = time.time()
            duration = stop - start
            print(duration*1000)

        print("times with prebuilt kdtree")
        xydata = data[:, :2]
        tree= kdtree.cKDTree(xydata)
        for n in [10,50,100]:
            start = time.time()
            prepare_interpolation_data_kdtree(point, data, n=n, tree=tree)
            stop = time.time()
            duration = stop - start
            print(duration*1000)

    #test_distances_kdtree()


    def test_distances():
        import test_data.point as p
        import os.path
        import time
        #from pysdss.geostat.variance import distance

        from scipy.spatial.distance import cdist
        from scipy.spatial.distance import pdist

        data = read_data(os.path.dirname(os.path.realpath(__file__)) + "/test_data/necoldem.dat")

        xydata = data[:,:2] #take only x,y


        ##### pdist
        start = time.time()
        dists = pdist(xydata)
        stop = time.time()
        duration = stop - start
        print(duration * 1000)

        #print(len(dists))
        #print(dists)

        ##### cdist (use this to get also the distance zero)
        start = time.time()
        dists = cdist(xydata,xydata)
        stop = time.time()
        duration = stop - start
        print(duration * 1000)

        print(dists.shape)
        print(dists)


        ###### with lists (slower)
        '''
        n = len(data)
        # we calculate the distances between pair of points
        start = time.time()
        dists2 = [[distance(xydata[i], xydata[j]) for i in range(n)] for j in range(n)]
        stop = time.time()
        duration = stop - start
        print(duration * 1000)
        print(dists2)
        '''

    #test_distances()

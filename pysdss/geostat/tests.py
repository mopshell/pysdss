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


def prepare_interpolation_data(point, data, n=10):
    """Get the nearest points to a point
    :param point: a point object
    :param data: a list of lists [[x,y,value],....]
    :param n: number of nearest points to the point x
    :return: the n nearest points to point and the average value of data
    """
    vals = [z[2] for z in data]
    mean = sum(vals)/len(vals)
    dist = [sqrt((z[0]-point.x)**2 + (z[1]-point.y)**2) for z in data]
    outdata = [(data[i][0], data[i][1], data[i][2], dist[i]) for i in range(len(dist))]
    outdata.sort(key=lambda x: x[3])
    outdata = outdata[:n]
    return outdata, mean


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
        import numpy as np
        from bokeh.plotting import figure
        from bokeh.io import output_file,  save
        from bokeh.models import Label
        import variance as vrc
        import fit

        data = read_data("./test_data/necoldem.dat")

        hh = 50 #half of the bin size
        lags = np.arange(0, 3000, hh)
        gamma = vrc.semivar(data, lags, hh)
        cov = vrc.covar(data,lags,hh)
        #print(gamma)
        #print(cov)

        #chart empirical semivariance and covariance
        output_file("./test_data/necoldem.dat.html")
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
        output_file("./test_data/necoldem.dat.fitted.html")
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

    # testing ordinary kriging for 2 points
    def test_ordinaryk():

        import numpy as np
        import variance as vrc
        import fit
        import test_data.point as p
        import kriging as kr

        data = read_data("./test_data/necoldem.dat")

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

        print("test point 337000, 4440911 ")
        x = p.Point(337000, 4440911)
        p1 = prepare_interpolation_data(x, data)[0]
        print(kr.ordinary(np.array(p1), semivariogram)[0:2])

        print("test a point from the original data")
        x = p.Point(data[1, 0], data[1, 1])
        p1 = prepare_interpolation_data(x, data)[0]
        print(kr.ordinary(np.array(p1), semivariogram)[0:2])

    test_ordinaryk()





# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:        feature_extract.py
# Purpose:     fuctions to extract features from the images
#
# Author:      Claudio Piccinini
#
# Created:     03/03/2017, revised 24/06/2014
# -------------------------------------------------------------------------------


import math
import cv2
import numpy as np
from osgeo import gdal_array as gdar
from osgeo import gdal


def makelines(path,algo="HoughLinesP", param={} ):


    d = gdal.Open(path)
    band = d.GetRasterBand(1)
    image = gdar.BandReadAsArray(band)
    d=None

    # set to uint8
    imagebw = image.astype(np.uint8, copy=False)

    #define the output image
    cdst = cv2.cvtColor(imagebw, cv2.COLOR_GRAY2BGR)


    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    # cv2.imshow("Blurred", image)
    # canny = cv2.Canny(image, 30, 150)
    # cv2.imshow("Canny", canny)
    # cv2.waitKey(0)


    if (algo=="HoughLinesP"):  # HoughLinesP

        if not param:
            lines = cv2.HoughLinesP(imagebw, 1, math.pi / 180.0, 40, np.array([]), 50, 10)
        else:
            lines = cv2.HoughLinesP(imagebw, param['rho'], param['theta'], param['threshold'],
                                    np.array([]), param['minlinelenght'], param['maxlinegap'])

        #print(lines)
        a, b, c = lines.shape
        print(lines.shape)
        for i in range(a):
            cv2.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3,
                     cv2.LINE_AA)

        cv2.imshow("detected lines", cdst)
        cv2.imwrite(path + "traced.jpeg",cdst )


    else:  # HoughLines
        if not param:
            lines = cv2.HoughLines(imagebw, 1, math.pi / 180.0, 50, np.array([]), 0, 0)
        else:
            lines = cv2.HoughLines(imagebw, param['rho'], param['theta'], param['threshold'],np.array([]), param['srn'], param['stn'])


        #print(lines)
        if lines is not None:
            a, b, c = lines.shape
            print(lines.shape)
            for i in range(a):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0, y0 = a * rho, b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow("detected lines", cdst)
            cv2.imwrite(path + "traced.jpeg",cdst)
    # cv2.imshow("source", src)

    cv2.waitKey(0)



if __name__ == '__main__':

    folder = r"D:\newcastle\vagrant\ubuntu64_webadmin\code\pysdss\data\output\text\rows"

    paths = ["/a4cc04d563b944d769a1513842356ed36_keep_rasterized_truth.tiff" ,
             "/a4cc04d563b944d769a1513842356ed36_keep_rasterized_step2.tiff",
             "/a4cc04d563b944d769a1513842356ed36_keep_rasterized_step3.tiff",
             "/a4cc04d563b944d769a1513842356ed36_keep_rasterized_step4.tiff",]

    param = {'rho':1, 'theta':math.pi / 180, 'threshold':40, 'minlinelenght':50, 'maxlinegap':200}

    param2 = {'rho':1, 'theta':math.pi / 360, 'threshold':100, 'srn':0, 'stn':0}

    for path in paths:
        makelines(folder+path, param=param2, algo="")

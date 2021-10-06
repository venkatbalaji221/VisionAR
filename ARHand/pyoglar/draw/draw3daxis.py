# /********************************************************************************************/
#   Written by: A K Dash
#   checked Env:  Py2Cv3 My Ubuntu 18.04
#   Description:
#           Draws a 3D axis on the aruco marker
#         - Aruco dependencies: python-aruco generated from source. (py2cv3 env on my system)
#
# ToDo
#      - Handle the Y-axis perpendicular or not results
#      - Check the axis directions
# /********************************************************************************************/
from __future__ import division
import cv2
#import aruco
import numpy as np


def draw3DAxis(frame, cameraMatix, distCoeff, Rvec, Tvec, axisSize, setYperpendicular=False):
    if setYperpendicular:
        objectPoints = np.array([
            [0, 0, 0],
            [axisSize, 0, 0],
            [0, axisSize, 0],
            [0, 0, axisSize]
        ], dtype=np.float32)
    else:
        objectPoints = np.array([
            [0, 0, 0],
            [axisSize, 0, 0],
            [0, 0, axisSize],
            [0, axisSize, 0]
        ], dtype=np.float32)

    imagePoints, jacobian = cv2.projectPoints(
        objectPoints, Rvec, Tvec, cameraMatix, distCoeff)
    int_imagepoints = imagePoints.astype(int)
    # print(imagePoints[0])
    # origin to x-axis
    cv2.line(frame, tuple(int_imagepoints[0].ravel()),
             tuple(int_imagepoints[1].ravel()), (0, 0, 255), 1, cv2.LINE_AA)
    # origin to y-axis
    cv2.line(frame, tuple(int_imagepoints[0].ravel()),
             tuple(int_imagepoints[2].ravel()), (0, 255, 0), 1, cv2.LINE_AA)
    # origin to z-axis
    cv2.line(frame, tuple(int_imagepoints[0].ravel()),
             tuple(int_imagepoints[3].ravel()), (255, 0, 0), 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'x', tuple(
        int_imagepoints[1].ravel()), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'y', tuple(
        int_imagepoints[2].ravel()), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'z', tuple(
        int_imagepoints[3].ravel()), font, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

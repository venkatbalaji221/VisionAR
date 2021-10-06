# /**************************************/
#   Written by: A K Dash
#   checked Env:  Py2Cv3 My Ubuntu 18.04
#   Description:
#           Draws a cube on the aruco marker
#         - Aruco dependencies: python-aruco generated from source. (py2cv3 env on my system)
#
# ToDo
#      - for the time being no more updates
# /************************************/
from __future__ import division
import cv2
#import aruco
import numpy as np


def draw3DCube(frame, cameraMatix, distCoeff, Rvec, Tvec, sideLen, setYperpendicular=True):
    '''
    @param:
        sideLength in meters
    '''
    halfSize = sideLen/2.0
    if setYperpendicular:
        objectPoints = np.array([[-halfSize, 0, -halfSize],
                                 [halfSize, 0, -halfSize],
                                 [halfSize, 0, halfSize],
                                 [-halfSize, 0, halfSize],
                                 [-halfSize, sideLen, -halfSize],
                                 [halfSize, sideLen, -halfSize],
                                 [halfSize, sideLen, halfSize],
                                 [-halfSize, sideLen, halfSize]
                                 ], dtype=np.float32)
    else:
        objectPoints = np.array([[-halfSize, -halfSize, 0],
                                 [halfSize, -halfSize, 0],
                                 [halfSize, halfSize, 0],
                                 [-halfSize, halfSize, 0],
                                 [-halfSize,  -halfSize, sideLen],
                                 [halfSize, -halfSize, sideLen],
                                 [halfSize, halfSize, sideLen],
                                 [-halfSize, halfSize, sideLen, ]
                                 ], dtype=np.float32)

    imagePoints, jacobian = cv2.projectPoints(
        objectPoints, Rvec, Tvec, cameraMatix, distCoeff)

    # Draw lines of different colors
    for i in range(4):
        cv2.line(frame, tuple(imagePoints[i].ravel()), tuple(imagePoints[(
            i+1) % 4].ravel()), (0, 255, 255), 2, cv2.LINE_AA)

    for i in range(4):
        cv2.line(frame, tuple(imagePoints[i+4].ravel()), tuple(imagePoints[4+(
            i+1) % 4].ravel()), (255, 0, 255), 2, cv2.LINE_AA)

    for i in range(4):
        cv2.line(frame, tuple(imagePoints[i].ravel()), tuple(
            imagePoints[i+4].ravel()), (255, 255, 0), 2, cv2.LINE_AA)

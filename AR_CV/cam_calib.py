###
# Author: Ajaya Kumar Dash
# Usage: python cam_calib.py --dirpath <Path to the directory containing chess-board images>
#
# References to Read:
#   1)  Camera Calibration:
#       - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
#   2)  Pose Estimation
#       - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html
#
####

import cv2
import numpy as np
import glob
# import yaml
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--dirpath", required=False,
                help="provide the path to the directory containing checker board image files")
args = vars(ap.parse_args())

# Term Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare the object points ( Real world coordinates considering the checker board as the xy-plane and z=0)
# (0, 0, 0), (1, 0, 0), (2, 0, 0) . . . (8, 0, 0)
# (0, 1, 0), (1, 1, 0), (2, 1, 0) . . . (8, 1, 0)
# (0, 2, 0), (1, 2, 0), (2, 2, 0) . . . (8, 2, 0)
#       .
#       .
#       .
# (0, 6, 0), (1, 6, 0), (2, 6, 0) . . . (8, 6, 0)
##

objp = np.zeros((9 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)*20
print(objp)

# Arrays to store the object points and image points from all the captured images
objpoints = []  # 3d point in the real world space
imgpoints = []  # 2d point in the image plane

image_path = args["dirpath"] + "/*.jpg"

images = glob.glob(image_path)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)
    if corners is not None:
        print("Type of corners:{}, Shape corner: {}, \nCorner:\n{}".format(type(corners), corners.shape, corners))

    # If found add the object points and image points (after refining them)
    if ret is True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw Chess baord corners for visualization purpose
        cv2.drawChessboardCorners(img, (9, 7), corners, ret)
        cv2.imshow("Chess Board Corners", img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the calibration parameters into a .npz file
np.savez("../simple/data/CalibData.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Reprojection Error
mean_error = 0
for i in np.arange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Calibration is done!!")
print("mean error: {} ".format(mean_error / len(objpoints)))

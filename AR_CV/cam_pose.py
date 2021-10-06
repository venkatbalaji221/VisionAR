###
# Author: Ajaya Kumar Dash
# Usage: python cam_pose.py
#
# References to Read:
#   1)  Camera Calibration:
#       - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
#   2)  Pose Estimation
#       - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html
#
####

import numpy as np
import cv2

# Term Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chessBoardSize = (9, 7)


def main():
    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        print("[Error] Opening the camera")
        exit(-1)

    cap.set(3, 640)  # CV_CAP_PROP_FRAME_WIDTH : Flag number 3
    cap.set(4, 480)  # CV_CAP_PROP_FRAME_HEIGHT : Flag number 4

    # Prepare the object points ( Real world coordinates considering the checker board as the xy-plane and z=0)
    # (0, 0, 0), (1, 0, 0), (2, 0, 0) . . . (8, 0, 0)
    # (0, 1, 0), (1, 1, 0), (2, 1, 0) . . . (8, 1, 0)
    # (0, 2, 0), (1, 2, 0), (2, 2, 0) . . . (8, 2, 0)
    #       .
    #       .
    #       .
    # (0, 6, 0), (1, 6, 0), (2, 6, 0) . . . (8, 6, 0)

    objp = np.zeros(
        (chessBoardSize[0] * chessBoardSize[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:chessBoardSize[0],
                           0:chessBoardSize[1]].T.reshape(-1, 2)

    # Load the intrinsic paremeters from the saved file from the calibration step.
    intrinsicCalibData = np.load("../ARHand/config/MyWebcamCalibData.npz")
    # for i in intrinsicCalibData.files:
    #     #print("{}\n\t{}".format(i, intrinsicCalibData[i]))
    #     print(i)
    # Output
    #       ['tvecs', 'mtx', 'dist', 'rvecs']

    intrinsicMatrix = intrinsicCalibData["mtx"]
    distCoeffs = intrinsicCalibData["dist"]

    flag = False

    # Loop over the receive the camera stream
    while True:
        ret, frame = cap.read()
        if ret is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            flag, corners = cv2.findChessboardCorners(gray, (9, 7), None)
            if flag is True:
                cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                # Draw Chess baord corners for visualization purpose.
                # The following step is not required during the pose estimation. However a good breaking point to
                # check your code segment.
                cv2.drawChessboardCorners(frame, chessBoardSize, corners, ret)
                retval, rvec, tvec = cv2.solvePnP(
                    objp, corners, intrinsicMatrix, distCoeffs)  # Use solvePnP() method to calculate the extrinsics -> rvecs and tvecs
                print(objp.shape)
                print(corners.shape)
                # print("rvec shape: {}, tvec Shape: {}".format(
                #     rvec.shape, tvec.shape))  # output 3x1 and 3x1. # type=float64,
                ## print("rvec_type={}, rvec:\n{}".format(rvec.dtype, rvec))
                ## print("tvec_type={}, rvec:\n{}".format(tvec.dtype, tvec))

                # Hence we use cv2.Rodrigues(..) to convert the rotation vector into rotation matrix (3 x3)
                # Convert rvec from float64 into float32 before using cv2.Rodrigues(..)

                rot_matrix, jacob = cv2.Rodrigues(np.float32(rvec))
                # or rot_matrix = cv2.Rodrigues(rvec)[0]
                # Append the tvec to the right side of the rotation matrix: camera pose matrix size 3x4
                extrinsicMatrix = np.hstack((rot_matrix, np.float32(tvec)))
                print("Extrinsic datatype = {}\n Extrinsic mat value = \n{}".format(
                    extrinsicMatrix.dtype, extrinsicMatrix))

            cv2.imshow("[Live Feed] Chess Board Corners", frame)
            # Press Q or Esc key from the keyboard to stop recording
            # witKey(..) returns an integer. odr(..) takes a single character and returns the equivalent unicode val
            c = cv2.waitKey(1)
            if c == 27 or c == ord('q'):
                break
        else:
            print("[Error] Reading the camera frame..")
            break


if __name__ == "__main__":
    main()

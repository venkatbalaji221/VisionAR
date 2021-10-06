# *****************************************************************************
# * Filename : ARonChessBoard.py
# * Breif Description : Implements AR on a chess board using openCV functions
# * Works with : Python 2 and opencv 3
# * Usage: $ python AR_Cube_onChessBoard.py
# *****************************************************************************


import cv2
import numpy as np

# Term Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
chessBoardSize = (9, 7)


def draw_cube(frame, x, y):
    cv2.line(frame, (int(x[0]), int(y[0])),
             (int(x[1]), int(y[1])), (255, 0, 0), 5)  # Represents x-axis edge(Blue)
    cv2.line(frame, (int(x[1]), int(y[1])),
             (int(x[2]), int(y[2])), (0, 128, 128), 5)
    cv2.line(frame, (int(x[2]), int(y[2])),
             (int(x[3]), int(y[3])), (0, 128, 128), 5)
    cv2.line(frame, (int(x[3]), int(y[3])),
             (int(x[0]), int(y[0])), (0, 255, 0), 5)  # Represents z-axis edge(Green)
    cv2.line(frame, (int(x[0]), int(y[0])),
             (int(x[4]), int(y[4])), (0, 0, 255), 5)  # Represents y-axis edge(Red)
    cv2.line(frame, (int(x[4]), int(y[4])),
             (int(x[5]), int(y[5])), (0, 128, 128), 5)
    cv2.line(frame, (int(x[5]), int(y[5])),
             (int(x[1]), int(y[1])), (0, 128, 128), 5)
    cv2.line(frame, (int(x[2]), int(y[2])),
             (int(x[6]), int(y[6])), (0, 128, 128), 5)
    cv2.line(frame, (int(x[5]), int(y[5])),
             (int(x[6]), int(y[6])), (0, 128, 128), 5)
    cv2.line(frame, (int(x[6]), int(y[6])),
             (int(x[7]), int(y[7])), (0, 128, 128), 5)
    cv2.line(frame, (int(x[7]), int(y[7])),
             (int(x[3]), int(y[3])), (0, 128, 128), 5)
    cv2.line(frame, (int(x[7]), int(y[7])),
             (int(x[4]), int(y[4])), (0, 128, 128), 5)


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

    intrinsicMatrix = np.float32(intrinsicCalibData["mtx"])
    distCoeffs = np.float32(intrinsicCalibData["dist"])

    cube_vertices = np.array([
        [0, 0, 0],
        [3, 0, 0],
        [3, 0, -3],
        [0, 0, -3],
        [0, 3, 0],
        [3, 3, 0],
        [3, 3, -3],
        [0, 3, -3]
    ], dtype=np.float32)
    one_append = np.ones((8, 1), dtype=np.float32)
    final_cube_vertices = np.hstack((cube_vertices, one_append))

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
                rot_matrix, jacob = cv2.Rodrigues(np.float32(rvec))
                # or rot_matrix = cv2.Rodrigues(rvec)[0]
                # Append the tvec to the right side of the rotation matrix: camera pose matrix size 3x4
                extrinsicMatrix = np.hstack((rot_matrix, np.float32(tvec)))
                cameraMatrix = np.dot(intrinsicMatrix, extrinsicMatrix)
                # print("Shape of camera matrix: {}".format(cameraMatrix.shape)) # output 3x4
                # Then we calculate the projection of 3D points on the image plane of the camera. Output is in homogeneous coordinate system
                homog_coord = np.dot(cameraMatrix, final_cube_vertices.T)
                x = homog_coord[0] / homog_coord[-1]  # x /
                # and y carries the coordinates for the four points of the axes
                y = (homog_coord[1] / homog_coord[-1])

                draw_cube(frame, x, y)

                # we then draw those ---------------------------------------------------------
                # cv2.line(frame, (int(x[0]), int(y[0])),
                #          (int(x[1]), int(y[1])), (255, 0, 0), 5)
                # cv2.line(frame, (int(x[1]), int(y[1])),
                #          (int(x[2]), int(y[2])), (0, 255, 0), 5)
                # cv2.line(frame, (int(x[2]), int(y[2])),
                #          (int(x[3]), int(y[3])), (0, 0, 255), 5)
                # cv2.line(frame, (int(x[3]), int(y[3])),
                #          (int(x[0]), int(y[0])), (0, 0, 255), 5)
                # cv2.line(frame, (int(x[0]), int(y[0])),
                #          (int(x[4]), int(y[4])), (0, 0, 255), 5)
                # cv2.line(frame, (int(x[4]), int(y[4])),
                #          (int(x[5]), int(y[5])), (0, 0, 255), 5)
                # cv2.line(frame, (int(x[5]), int(y[5])),
                #          (int(x[1]), int(y[1])), (0, 0, 255), 5)
                # cv2.line(frame, (int(x[2]), int(y[2])),
                #          (int(x[6]), int(y[6])), (0, 0, 255), 5)
                # cv2.line(frame, (int(x[5]), int(y[5])),
                #          (int(x[6]), int(y[6])), (0, 0, 255), 5)
                # cv2.line(frame, (int(x[6]), int(y[6])),
                #          (int(x[7]), int(y[7])), (0, 0, 255), 5)
                # cv2.line(frame, (int(x[7]), int(y[7])),
                #          (int(x[3]), int(y[3])), (0, 0, 255), 5)
                # cv2.line(frame, (int(x[7]), int(y[7])),
                #          (int(x[4]), int(y[4])), (0, 0, 255), 5)

            cv2.imshow("[Live Feed] Chess Board Corners", frame)
            # Press Q or Esc key from the keyboard to stop recording
            # witKey(..) returns an integer. odr(..) takes a single character and returns the equivalent unicode val
            c = cv2.waitKey(1)
            if c == 27 or c == ord('q'):
                return -1


if __name__ == "__main__":
    main()

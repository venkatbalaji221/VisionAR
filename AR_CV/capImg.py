#####
# This code is for capturing the checker board images from a single camera video stream for calibration
#
# Author: Ajaya Kumar Dash
# Usage : $ python cap_img_single_cam.py --dirname <Directory Name with path to store the captured image files>
#
#
# REFERENCES To Read:
# 1) Video Capture Object: get and set
#       - https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
# 2) Python os.path module related
#       - https://docs.python.org/3/library/os.path.html#module-os.path
#
####

import cv2
import numpy as np
import os
import argparse


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


def main():
    cap = cv2.VideoCapture(4)
    if cap.isOpened() is False:
        print("Error: Opening the camera")
        exit(-1)

    cap.set(3, 640)  # CV_CAP_PROP_FRAME_WIDTH : Flag number 3
    cap.set(4, 480)  # CV_CAP_PROP_FRAME_HEIGHT : Flag number 4
    print("Press 'c' to capture a frame and 'q to quit the video stream from the camera..")

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dirname", required=True,
                    help="Provide the directory name to stored the captured images for calibration")
    args = vars(ap.parse_args())

    createFolder(args["dirname"])

    choice = ord('z')
    count = 0

    while choice is not ord('q'):
        ret, frame = cap.read()
        if ret is True:
            # frame = cv2.resize(frame, (0, 0), fx= 2, fy= 2)
            cv2.imshow("Live Video", frame)
            if choice is ord('c'):
                file_name = "calib_img" + str(count) + ".jpg"
                cv2.imwrite(os.path.join(args["dirname"], file_name), frame)
                count = count + 1
            choice = cv2.waitKey(1)
        else:
            print("Unable to read the frames from the camera stream!!")
            break


if __name__ == "__main__":
    main()

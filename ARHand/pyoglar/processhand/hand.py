# /**************************************/
#   Written by: A K Dash
#   checked Env:  Py3Cv3 My Ubuntu 18.04
#   Description:
#         - Detect the Media pipe hand
#
#
# ToDo
#      -
#      -
# /************************************/

import os
import sys
import numpy as np
import cv2
import glm
import pkg_resources  # part of python setup tool
import mediapipe as mp
mp_drawings = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class Hand:
    '''
        Params: (<path to camera config file>, <path to board config file>)
    '''

    def __init__(self, pathcamparamfile=None, pathboardconfigfile=None):

        self.pathcamparamfile = pathcamparamfile
        self.pathboardconfigfile = pathboardconfigfile

    def init_camparams(self):
        fileName, fileExtension = os.path.splitext(self.pathcamparamfile)
        # print("FileName: {}".format(fileName))
        # print("FileExtension: {}".format(fileExtension))
        if fileExtension == ".yml":
            # [ToDo] : write your own yml parser.
            pass
            # camparam = aruco.CameraParameters()
            # # Contains camera intrinsic params : camera matrix and distortion coefficient
            # camparam.readFromXMLFile(self.pathcamparamfile)
            # self.CameraMatrix = camparam.CameraMatrix
            # self.Distorsion = camparam.Distorsion
        elif fileExtension == ".npz":
            camparam = np.load(self.pathcamparamfile)
            self.CameraMatrix = camparam["mtx"]
            self.Distorsion = camparam["dist"]

    def handDetector(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mp_hands.Hands(
            max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    def handLandmarkDetect(self, frame, handNumber=0, draw=False):
        #originalFrame = frame
        # Mediapipe needs RGB
        self.hand_detect_flag = False
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = self.hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:  # Returns None if hand is not found
            # returns landmarks for all the hands
            self.hand_detect_flag = True
            self.handlandmarks = results.multi_hand_landmarks

        if draw:
            pass

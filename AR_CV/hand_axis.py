import numpy as np
import cv2
import mediapipe as mp
import time 

start = time.time()
prev_frame_time = start
new_frame_time = 0

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def getImgp(i,hand_landmarks, img_wd, img_ht):
    x = int(hand_landmarks.landmark[i].x * img_wd)
    y = int(hand_landmarks.landmark[i].y * img_ht)
    mcp_point = [x, y]
    return mcp_point

def main():
    global prev_frame_time, new_frame_time
    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        print("[Error] Opening the camera")
        exit(-1)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    out = cv2.VideoWriter('output1.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
    objp = np.zeros((6, 3), dtype=np.float32)
    # objp[:, :2] = [[69, 118], [92, 119], [118, 122],[144, 134]]  # object points from graph paper(Right Hand)
    objp[:, :2] = [[68, 214], [104, 198], [109, 124], [85, 118], [62, 124], [40, 137]]   # Left Hand

    # Load the intrinsic paremeters from the saved file from the calibration step.
    intrinsicCalibData = np.load("../ARHand/config/MyWebcamCalibData.npz")

    intrinsicMatrix = intrinsicCalibData["mtx"]
    distCoeffs = intrinsicCalibData["dist"]

    vertices = np.float32(
        [[85, 118, 0], [145, 118, 0], [85, 178, 0], [85, 118, -60]]) # Axis coordinates with point-9 as origin

    axis = np.hstack(
        (vertices, np.array([[1], [1], [1], [1]], dtype=np.float32))) 
    flag = False

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            img_ht, img_wd, img_ch = image.shape
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=BLACK), mp_drawing.DrawingSpec(color=WHITE) )

                if not hand_landmarks.landmark[2].x < hand_landmarks.landmark[17].x:  # LeftHandedness
                    imgp = np.zeros((6, 2), dtype=np.float32)
                    k = 0
                    for i in [0, 1, 5, 9, 13, 17]:
                        result_point = getImgp(i, hand_landmarks, img_wd, img_ht)
                        imgp[k, :] = np.array(result_point, dtype= np.float32)
                        k = k + 1
                  
                    print("ObjectPoints:\n", objp)
                    print("ImagePoints:\n", imgp)

                    retval, rvec, tvec = cv2.solvePnP(objp, imgp, intrinsicMatrix, distCoeffs)
                    rot_matrix, jacob = cv2.Rodrigues(np.float32(rvec))
                    extrinsicMatrix = np.hstack((rot_matrix, np.float32(tvec)))
                    print("Extrinsic datatype = {}\nExtrinsic mat value = \n{}\n".format(
                        extrinsicMatrix.dtype, extrinsicMatrix))
                    cameraMatrix = np.dot(intrinsicMatrix, extrinsicMatrix)

                    homog_coord = np.dot(cameraMatrix, axis.T)
                    x = homog_coord[0] / homog_coord[-1]  
                    y = homog_coord[1] / homog_coord[-1]

                    cv2.line(image, (int(x[0]), int(y[0])),
                             (int(x[1]), int(y[1])), (255, 0, 0), 3)
                    cv2.line(image, (int(x[0]), int(y[0])),
                            (int(x[2]), int(y[2])), (0, 255, 0), 3)
                    cv2.line(image, (int(x[0]), int(y[0])),
                            (int(x[3]), int(y[3])), (0, 0, 255), 3)


            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            cv2.putText(image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            out.write(image)  # writing into video file
            image = cv2.resize(image, (0, 0), fx=2, fy=2)
            cv2.imshow("HandAxis", image)
            cv2.imwrite("augment.png", image)

            if cv2.waitKey(5) & 0xFF == 27: # checking if esc key is pressed
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  
    main()
    print( int(time.time() - start) )

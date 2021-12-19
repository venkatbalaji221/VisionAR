import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image
from glumpy import app, gl, glm, gloo, data, log
from utils import get_graphdata, getImgp, get_projection, get_view
from glumpy_utils import bgFrame

# Mediapipe initializers
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initial zoom parameter
base_zoom = 0.2
# queues to store data of adjacent frames for zooming and registering object
zoom_queue, reg_queue = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
# Rotation angle around y and x axis
phi, theta = 5, 5
# Register flag ( Not registered = 0, object cleared = -1, Registered = 1)  and Rotation flag (Around Y - 1, X - 2)
register, rotate_mode = 0, 0
# Point in space where object needs to be registered
base_point = [0, 0, 0]

with open("../shaders/vertex.vert", 'r') as f:  # V shader for cube
    vertex = f.read()
with open("../shaders/fragment.frag", 'r') as f:  # F shader for cube
    fragment = f.read()
with open("../shaders/vertex1.vert", 'r') as f:  # V shader for background
    vertex1 = f.read()
with open("../shaders/fragment1.frag", 'r') as f:  # F shader for background
    fragment1 = f.read()

window = app.Window(width=1920, height=1080, color=(1, 1, 1, 1))


@window.event
def on_draw(dt):
    global base_point, base_zoom, zoom_queue, reg_queue, phi, theta, register, rotate_mode
    window.clear()
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Draw BG_Texture
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Rotate around y-axis
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        img_ht, img_wd, img_ch = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                # Drawing hand skeleton for detected hands
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=BLACK), mp_drawing.DrawingSpec(color=WHITE))
            # LeftHandedness for registering object
            if hand_landmarks.landmark[2].x > hand_landmarks.landmark[17].x:
                if register == 0 or register == -1:
                    register = 0
                    imgp = np.zeros((objp.shape[0], 2), dtype=np.float32)
                    k = 0
                    for i in [0, 1, 5, 9, 13, 17]:
                        result_point = getImgp(i, hand_landmarks, img_wd, img_ht)
                        imgp[k, :] = np.array(result_point, dtype=np.float32)
                        k = k + 1
                        # selecting keypoint-5 manually to place object on the hand
                        if i == 5:
                            base_point[0:2] = result_point
                            scale_mat = glm.scale(np.eye(4, dtype=np.float32), base_zoom, base_zoom, base_zoom)
                            globj['model'] = glm.translate(scale_mat, base_point[0], base_point[1], base_point[2])

                    retval, rvec, tvec = cv2.solvePnP(objp, imgp, intrinsicMatrix, distCoeffs)
                    rot_matrix, jacob = cv2.Rodrigues(np.float32(rvec))
                    extrinsicMatrix = np.hstack((rot_matrix, np.float32(tvec)))
                    view_mat = get_view(extrinsicMatrix)
                    globj['view'] = view_mat
                    # print('{0} \n\n'.format(extrinsicMatrix))
            # RightHandedness for Gestures
            else:
                # Detecting status of each finger whether raised or closed
                thumbIsOpen = False
                firstFingerIsOpen = False
                secondFingerIsOpen = False
                thirdFingerIsOpen = False
                fourthFingerIsOpen = False

                anchorPoint = hand_landmarks.landmark[2].x
                if hand_landmarks.landmark[3].x < anchorPoint and hand_landmarks.landmark[4].x < anchorPoint:
                    thumbIsOpen = True

                anchorPoint = hand_landmarks.landmark[6].y
                if hand_landmarks.landmark[7].y < anchorPoint and hand_landmarks.landmark[8].y < anchorPoint:
                    firstFingerIsOpen = True

                anchorPoint = hand_landmarks.landmark[10].y
                if hand_landmarks.landmark[11].y < anchorPoint and hand_landmarks.landmark[12].y < anchorPoint:
                    secondFingerIsOpen = True

                anchorPoint = hand_landmarks.landmark[14].y
                if hand_landmarks.landmark[15].y < anchorPoint and hand_landmarks.landmark[16].y < anchorPoint:
                    thirdFingerIsOpen = True

                anchorPoint = hand_landmarks.landmark[18].y
                if hand_landmarks.landmark[19].y < anchorPoint and hand_landmarks.landmark[20].y < anchorPoint:
                    fourthFingerIsOpen = True

                hand_gesture = [thumbIsOpen, firstFingerIsOpen, secondFingerIsOpen, thirdFingerIsOpen,
                                fourthFingerIsOpen]

                # pointFinger flag (True - Raised, False - Closed)
                if hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y:
                    pointFinger = True
                else:
                    pointFinger = False

                # Clearing Object
                if hand_gesture == [False, True, False, False, True]:
                    if register == 1:
                        reg_queue.pop(0)
                        reg_queue.append(0)
                    if sum(reg_queue) == 0:
                        register = -1
                        window.clear()
                        base_zoom = 0.2

                # Registering Object
                if hand_gesture == [True, True, False, False, True]:
                    if register == 0:
                        reg_queue.pop(0)
                        reg_queue.append(1)
                    if sum(reg_queue) == 5:
                        register = 1

                thumb_x, thumb_y = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
                index_x, index_y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y

                # Ring and Little Fingers Closed, pointFinger raised for both zoom in and out
                if hand_gesture[3:] == [False, False] and pointFinger:
                    length = math.hypot(index_x - thumb_x, index_y - thumb_y)
                    zoom_queue.pop(0)
                    zoom_queue.append(length)
                    # Middle Finger closed for zoom-in
                    if hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y:
                        # check if values in the queue are ascending
                        if all(zoom_queue[i] <= zoom_queue[i + 1] for i in range(len(zoom_queue) - 1)):
                            base_zoom += length / 20
                    else:
                        # Middle Finger raised for zoom-out
                        # check if values in the queue are descending
                        if all(zoom_queue[i] >= zoom_queue[i + 1] for i in range(len(zoom_queue) - 1)):
                            if base_zoom > length / 15:
                                base_zoom -= length / 15

                    # Applying zoom-in or zoom-out finally after computing base_zoom
                    model = np.eye(4, dtype=np.float32)
                    glm.scale(model, base_zoom, base_zoom, base_zoom)
                    glm.translate(model, base_point[0], base_point[1], base_point[2])
                    globj['model'] = model.reshape([16, ])

                # Rotating object
                if hand_gesture == [False, True, True, True, True]:
                    rotate_mode = 1  # Rotate around Y - axis
                elif hand_gesture == [True, False, False, False, False] \
                        and hand_landmarks.landmark[4].y < hand_landmarks.landmark[8].y:
                    rotate_mode = 2  # Rotate around X - axis
                else:
                    rotate_mode = 0

                if rotate_mode in [1, 2]:
                    model = globj['model'].reshape([4, 4])
                    glm.translate(model, -base_point[0], -base_point[1], -base_point[2])
                    if rotate_mode == 1:  # Y-axis
                        phi += 0.01
                        glm.rotate(model, phi, 0, 1, 0)
                    elif rotate_mode == 2:  # X-axis
                        theta += 0.01
                        glm.rotate(model, theta, 1, 0, 0)
                    glm.translate(model, base_point[0], base_point[1], base_point[2])
                    globj['model'] = model.reshape([16, ])

    frame = cv2.flip(frame, 0)  # Rotate around x-axis
    cv2.imwrite("frame.jpg", frame)
    bgTex['texture'] = np.array(Image.open("./frame.jpg"))
    # frame = cv2.resize(frame, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("Live feed", cv2.flip(frame, 0))
    cv2.waitKey(1)
    bgTex.draw(gl.GL_TRIANGLES, I1)
    # print(register)
    # Draw object
    if register != -1:
        globj.draw(gl.GL_TRIANGLES, indices)


@window.event
def on_resize(width, height):
    gl_projectionmat = get_projection(intrinsicMatrix, 2.0, 100.0)
    bgTex['perspective'] = gl_projectionmat
    globj['perspective'] = gl_projectionmat


@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glPolygonOffset(1, 1)
    gl.glEnable(gl.GL_LINE_SMOOTH)


if __name__ == "__main__":

    # Background Texture
    V1, I1 = bgFrame()
    bgTex = gloo.Program(vertex1, fragment1)
    bgTex.bind(V1)
    bgTex['model'] = np.eye(4, dtype=np.float32)
    bgTex['view'] = glm.translation(0, 0, -4)

    cap = cv2.VideoCapture(0)

    # Mediapipe code
    palm_points = get_graphdata()
    objp = np.zeros((len(palm_points), 3), dtype=np.float32)
    shifted_points = [[(x - 90) / 90, -(y - 120) / 120] for (x, y) in palm_points]  # (90, 120) is the center in graph paper
    objp[:, :2] = shifted_points  # Left Hand
    intrinsicCalibData = np.load("../data/CalibData.npz")
    intrinsicMatrix = intrinsicCalibData["mtx"]
    distCoeffs = intrinsicCalibData["dist"]

    # Object Rendering
    with open("./objShaders/vertobj.vert", 'r') as f:  # V shader for background
        vertexObj = f.read()
    with open("./objShaders/fragobj.frag", 'r') as f:  # F shader for background
        fragmentObj = f.read()

    objfile = sys.argv[1]
    log.info("Loading Object!")
    local_dir = os.path.dirname(os.path.abspath(__file__))
    local_file = local_dir + "/" + objfile
    vertices, indices = data.get(local_file)
    vertices = vertices.view(gloo.VertexBuffer)
    indices = indices.view(gloo.IndexBuffer)
    globj = gloo.Program(vertexObj, fragmentObj)
    globj.bind(vertices)

    if len(sys.argv) == 3:
        texfile = sys.argv[2]
        globj['texture'] = np.array(Image.open(texfile))
        globj['texture'].interpolation = gl.GL_LINEAR
        globj['tex'] = 1

    app.run(framerate=30)
    cv2.destroyAllWindows()
    cap.release()

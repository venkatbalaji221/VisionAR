import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image
from glumpy import app, gl, glm, gloo, data, log
from glumpy.transforms import Trackball, Position
from utils import get_graphdata, getImgp, get_projection, get_view
from glumpy_utils import bgFrame, update, Cube

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cameraMatrix, base_point = None, None
scale_mat = glm.scale(np.eye(4, dtype=np.float32), 0.2, 0.2, 0.2)
mini, maxi = 10, 0
base_zoom = 0.2
zoom_queue = [0, 0, 0, 0, 0]
reg_queue = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
phi, theta, zhi = 5, 5, 5
register = 0
windowAction = ""
with open("../shaders/vertex.vert", 'r') as f:  # V shader for cube
    vertex = f.read()
with open("../shaders/fragment.frag", 'r') as f:  # F shader for cube
    fragment = f.read()
with open("../shaders/vertex1.vert", 'r') as f:  # V shader for background
    vertex1 = f.read()
with open("../shaders/fragment1.frag", 'r') as f:  # F shader for background
    fragment1 = f.read()

# window = app.Window(width=1280, height=960, color=(1, 1, 1, 1))
window = app.Window(width=1920, height=1080, title=windowAction, color=(1, 1, 1, 1))

@window.event
def on_draw(dt):
    global cap, cameraMatrix, mini, maxi, base_point, base_zoom, zoom_queue, scale_mat, phi, theta, zhi, window, register, reg_queue, windowAction

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
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=BLACK), mp_drawing.DrawingSpec(color=WHITE))
            # LeftHandedness for rendering object
            if hand_landmarks.landmark[2].x > hand_landmarks.landmark[17].x and register == 0:
                imgp = np.zeros((objp.shape[0], 2), dtype=np.float32)
                k = 0
                for i in [0, 1, 5, 9, 13, 17]:
                    result_point = getImgp(i, hand_landmarks, img_wd, img_ht)
                    imgp[k, :] = np.array(result_point, dtype=np.float32)
                    k = k + 1
                    if i == 13:
                        base_point = result_point
                        cubeObj.cube['model'] = glm.translation(result_point[0] + 0.1, result_point[1] - 0.1, 0.1)
                        scale_mat = glm.scale(np.eye(4, dtype=np.float32), base_zoom, base_zoom, base_zoom)
                        globj['model'] = glm.translate(scale_mat, result_point[0] + 0.1, result_point[1] - 0.1, 0.1)
                retval, rvec, tvec = cv2.solvePnP(objp, imgp, intrinsicMatrix, distCoeffs)
                rot_matrix, jacob = cv2.Rodrigues(np.float32(rvec))
                extrinsicMatrix = np.hstack((rot_matrix, np.float32(tvec)))
                view_mat = get_view(extrinsicMatrix)
                cubeObj.cube['view'] = view_mat
                globj['view'] = view_mat
                # globj['m_normal'] = np.array(np.matrix(np.dot(globj['m_view'], globj['m_model'])).I.T)
                # print('{0} \n\n'.format(extrinsicMatrix))
            # RightHandedness for Gestures
            else:
                thumb_x, thumb_y = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
                index_x, index_y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
                # Ring Finger Closed for both zoom in and out
                if hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y:
                    length = math.hypot(index_x - thumb_x, index_y - thumb_y)
                    zoom_queue.pop(0)
                    zoom_queue.append(length)
                    # Middle Finger closed for zoom-in
                    if hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y:
                        if all(zoom_queue[i] <= zoom_queue[i + 1] for i in range(len(zoom_queue) - 1)):
                            base_zoom += length / 20
                            scale_mat = glm.scale(np.eye(4, dtype=np.float32), base_zoom, base_zoom, base_zoom)
                            globj['model'] = scale_mat
                            globj['model'] = glm.translate(scale_mat, base_point[0], base_point[1], 0.1)
                    else:
                        # Middle Finger raised for zoom-out
                        if all(zoom_queue[i] >= zoom_queue[i + 1] for i in range(len(zoom_queue) - 1)):
                            if base_zoom > length / 15:
                                base_zoom -= length / 15
                            scale_mat = glm.scale(np.eye(4, dtype=np.float32), base_zoom, base_zoom, base_zoom)
                            globj['model'] = scale_mat
                            globj['model'] = glm.translate(scale_mat, base_point[0], base_point[1], 0.1)

                # Thumsup gesture for rotating object
                min_tip = 1000
                for i in [8, 12, 16, 20]:
                    if min_tip > hand_landmarks.landmark[i].y:
                        min_tip = hand_landmarks.landmark[i].y

                if min_tip > hand_landmarks.landmark[3].y:
                    # globj.draw(gl.GL_TRIANGLES, indices)
                    theta += 0.01
                    phi -= 0.01
                    zhi -= 0.01
                    model = globj['model'].reshape([4, 4])
                    # glm.rotate(model, theta, 0, 0, 1)
                    glm.rotate(model, phi, 0, 1, 0)
                    # glm.rotate(model, zhi, 1, 0, 0)
                    globj['model'] = model.reshape([16, ])

                # For deregistering the object
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
                # Registering Index
                if hand_gesture == [True, True, False, False, True]:
                    if register == 0:
                        reg_queue.pop(0)
                        reg_queue.append(1)
                    if sum(reg_queue) == 10:
                        register = 1
                        windowAction = "Registered"
                # Deregistering Index
                if hand_gesture == [False, True, False, False, True]:

                    if register == 1:
                        reg_queue.pop(0)
                        reg_queue.append(0)
                    if sum(reg_queue) == 0:
                        register = 0
                        windowAction = "Free"

    frame = cv2.flip(frame, 0)  # Rotate around x-axis
    cv2.imwrite("frame.jpg", frame)
    bgTex['texture'] = np.array(Image.open("./frame.jpg"))
    frame = cv2.resize(frame, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("Live feed", cv2.flip(frame, 0))
    cv2.waitKey(1)
    bgTex.draw(gl.GL_TRIANGLES, I1)

    # Fill Cube
    # cubeObj.cube.draw(gl.GL_TRIANGLES, cubeObj.I)

    # Render Object
    globj.draw(gl.GL_TRIANGLES, indices)

    print(register)


@window.event
def on_resize(width, height):
    # cube['projection'] = glm.perspective(30.0, width / float(height), 2.0, 100.0)

    gl_projectionmat = get_projection(intrinsicMatrix, 2.0, 100.0)
    cubeObj.cube['perspective'] = gl_projectionmat
    bgTex['perspective'] = gl_projectionmat
    globj['perspective'] = gl_projectionmat


@window.event
def on_mouse_drag(x, y, dx, dy, button):
    update(trackball, globj)


@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    update(trackball, globj)
    gl.glPolygonOffset(1, 1)
    gl.glEnable(gl.GL_LINE_SMOOTH)


if __name__ == "__main__":

    # Cube object initialisation
    cubeObj = Cube(vertex, fragment)

    # Background Texture
    V1, I1 = bgFrame()
    bgTex = gloo.Program(vertex1, fragment1)
    bgTex.bind(V1)
    bgTex['model'] = np.eye(4, dtype=np.float32)
    bgTex['view'] = glm.translation(0, 0, -4)

    cap = cv2.VideoCapture(4)
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 75)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    # cap.set(cv2.CAP_PROP_CONTRAST, 1)
    # cap.set(cv2.CAP_PROP_BACKLIGHT, 10)

    # cap = cv2.VideoCapture(0)
    # Mediapipe code
    palm_points = get_graphdata()
    objp = np.zeros((len(palm_points), 3), dtype=np.float32)
    # objp[:, :2] = [[69, 118], [92, 119], [118, 122], [144, 134]]  # Right hand coordinates
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
    print(local_file)
    vertices, indices = data.get(local_file)
    print(indices)
    globj = gloo.Program(vertexObj, fragmentObj)
    globj.bind(vertices)

    trackball = Trackball(Position("position"))
    globj['transform'] = trackball
    trackball.theta, trackball.phi, trackball.zoom = 0, 0, 35
    if len(sys.argv) >= 3:
        texfile = sys.argv[2]
        globj['texture'] = np.array(Image.open(texfile))
        globj['texture'].interpolation = gl.GL_LINEAR
        globj['tex'] = 1
    else:
        print('hai')
        globj['tex'] = 0

    window.attach(globj['transform'])
    app.run()
    cv2.destroyAllWindows()
    cap.release()

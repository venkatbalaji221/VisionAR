import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from glumpy import app, gl, glm, gloo, data, log
from glumpy.transforms import Trackball, Position

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cameraMatrix = None


def get_graphdata():
    return [[68, 214], [104, 198], [109, 124], [85, 118], [62, 124], [40, 137]]
    # 0 1 5 9 13 17 - Hand Keypoint serial Numbers


def get_vertices(corner_x, corner_y):
    # corner_x = corner_x / 320
    # corner_y = corner_y / 240
    cube_vertices = np.array([
        [corner_x, corner_y, 0],
        [corner_x, corner_y + 0.2, 0],
        [corner_x + 0.2, corner_y + 0.2, 0],
        [corner_x + 0.2, corner_y, 0],
        [corner_x + 0.2, corner_y, 0.2],
        [corner_x, corner_y, 0.2],
        [corner_x, corner_y + 0.2, 0.2],
        [corner_x + 0.2, corner_y + 0.2, 0.2]
    ], dtype=np.float32)
    return cube_vertices


def getImgp(i, hand_landmarks, img_wd, img_ht):
    # if hand_landmarks is not None:
    #     print(hand_landmarks.landmark[i], end=' ')
    x = int(hand_landmarks.landmark[i].x * img_wd)
    y = int(hand_landmarks.landmark[i].y * img_ht)
    mcp_point = [(x - 320) / 320, -(y - 240) / 240]
    # mcp_point = [x, y]
    return mcp_point


def get_projection(camMatrix, n, f):
    print(camMatrix)
    fx = np.float32(camMatrix[0][0])
    fy = np.float32(camMatrix[1][1])
    cx = np.float32(camMatrix[0][2])
    cy = np.float32(camMatrix[1][2])
    # cx = 320
    # cy = 240
    persp_mat = np.array([[fx / cx, 0, 0, 0],
                          [0, fy / cy, 0, 0],
                          [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)],
                          [0, 0, -1, 0]])

    return np.transpose(persp_mat)


def get_view(em):
    # print(em)
    # mvmatrix = np.array([[em[0][0], em[0][1], em[0][2], em[0][3]],
    #                      [-em[1][0], -em[1][1], -em[1][2], -em[1][3]],
    #                      [-em[2][0], -em[2][1], -em[2][2], -em[2][3]],
    #                      [0, 0, 0, 1]])
    mvmatrix = np.array([[em[0][0], em[0][1], em[0][2], 0],
                         [em[1][0], em[1][1], em[1][2], 0],
                         [em[2][0], em[2][1], em[2][2], -3],
                         [0, 0, 0, 1]])
    view = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -3], [0, 0, 0, 1]])
    return np.transpose(mvmatrix)


def bgFrame():  # Texture for background
    vtype_frame = [('position', np.float32, 3),
                   ('texcoord', np.float32, 2)]
    itype = np.uint32
    # Vertices positions
    p = np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]], dtype=float)
    # Texture coords
    t = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    faces_p = [0, 1, 2, 3]
    faces_t = [0, 1, 2, 3]
    vertices = np.zeros(4, vtype_frame)
    vertices['position'] = p[faces_p]
    vertices['texcoord'] = t[faces_t]
    filled = np.resize(np.array([0, 1, 2, 0, 2, 3], dtype=itype), 1 * (2 * 3))
    # filled += np.repeat(4 * np.arange(6, dtype=itype), 6)
    vertices = vertices.view(gloo.VertexBuffer)
    filled = filled.view(gloo.IndexBuffer)
    return vertices, filled


with open("../shaders/vertex.vert", 'r') as f:  # V shader for cube
    vertex = f.read()
with open("../shaders/fragment.frag", 'r') as f:  # F shader for cube
    fragment = f.read()
with open("../shaders/vertex1.vert", 'r') as f:  # V shader for background
    vertex1 = f.read()
with open("../shaders/fragment1.frag", 'r') as f:  # F shader for background
    fragment1 = f.read()

window = app.Window(width=1280, height=960, color=(1, 1, 1, 1))


def update():
    model = trackball['model'].reshape(4, 4)
    view = trackball['view'].reshape(4, 4)
    projection = trackball['projection'].reshape(4, 4)
    brain['m_view'] = view
    brain['m_model'] = model
    brain['m_projection'] = projection
    brain['m_normal'] = np.array(np.matrix(np.dot(view, model)).I.T)


@window.event
def on_draw(dt):
    global cap, cameraMatrix, V

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
            if not hand_landmarks.landmark[2].x < hand_landmarks.landmark[17].x:
                imgp = np.zeros((objp.shape[0], 2), dtype=np.float32)
                k = 0
                for i in [0, 1, 5, 9, 13, 17]:
                    result_point = getImgp(i, hand_landmarks, img_wd, img_ht)
                    imgp[k, :] = np.array(result_point, dtype=np.float32)
                    k = k + 1
                    if i == 13:
                        cube['model'] = glm.translation(result_point[0] + 0.1, result_point[1] - 0.1, 0.1)
                        scale_mat = glm.scale(np.eye(4, dtype=np.float32), 0.3, 0.3, 0.3)
                        brain['model'] = glm.translate(scale_mat, result_point[0] + 0.1, result_point[1] - 0.1, 0.1)
                retval, rvec, tvec = cv2.solvePnP(objp, imgp, intrinsicMatrix, distCoeffs)
                rot_matrix, jacob = cv2.Rodrigues(np.float32(rvec))
                extrinsicMatrix = np.hstack((rot_matrix, np.float32(tvec)))
                view_mat = get_view(extrinsicMatrix)
                cube['view'] = view_mat
                brain['view'] = view_mat
                brain['m_normal'] = np.array(np.matrix(np.dot(brain['view'], brain['model'])).I.T)
                # print('{0} \n\n'.format(extrinsicMatrix))
            # RightHandedness for Gestures
            else:
                pass

    frame = cv2.flip(frame, 0)  # Rotate around x-axis
    cv2.imwrite("frame.jpg", frame)
    bgTex['texture'] = np.array(Image.open("./frame.jpg"))
    frame = cv2.resize(frame, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("Live feed", cv2.flip(frame, 0))
    cv2.waitKey(1)
    bgTex.draw(gl.GL_TRIANGLES, I1)

    # Fill Cube
    # cube.draw(gl.GL_TRIANGLES, I)

    # Render Object
    brain.draw(gl.GL_TRIANGLES, indices)


@window.event
def on_resize(width, height):
    # cube['projection'] = glm.perspective(30.0, width / float(height), 2.0, 100.0)

    gl_projectionmat = get_projection(intrinsicMatrix, 2.0, 100.0)
    cube['perspective'] = gl_projectionmat
    bgTex['perspective'] = gl_projectionmat
    brain['perspective'] = gl_projectionmat


@window.event
def on_mouse_drag(x, y, dx, dy, button):
    update()


@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    update()
    gl.glPolygonOffset(1, 1)
    gl.glEnable(gl.GL_LINE_SMOOTH)


if __name__ == "__main__":
    filepath = sys.argv[1]
    # Data for Cube
    V = np.zeros(8, [("position", np.float32, 3),
                     ("color", np.float32, 4)])
    V["position"] = np.array([[-0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0], [0.1, -0.1, 0],
                              [0.1, -0.1, 0.2], [-0.1, -0.1, 0.2], [-0.1, 0.1, 0.2], [0.1, 0.1, 0.2]], dtype=np.float32)
    V["color"] = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
                  [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]
    V = V.view(gloo.VertexBuffer)

    I = np.array([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 1,
                  1, 6, 7, 1, 7, 2, 7, 4, 3, 7, 3, 2, 4, 7, 6, 4, 6, 5], dtype=np.uint32)
    I = I.view(gloo.IndexBuffer)
    cube = gloo.Program(vertex, fragment)
    cube.bind(V)
    cube['model'] = np.eye(4, dtype=np.float32)

    # Data for background Texture
    V1, I1 = bgFrame()
    bgTex = gloo.Program(vertex1, fragment1)
    bgTex.bind(V1)
    bgTex['model'] = np.eye(4, dtype=np.float32)
    bgTex['view'] = glm.translation(0, 0, -3)

    cap = cv2.VideoCapture(0)

    # Mediapipe code
    palm_points = get_graphdata()
    objp = np.zeros((len(palm_points), 3), dtype=np.float32)
    # objp[:, :2] = [[69, 118], [92, 119], [118, 122], [144, 134]]  # Right hand coordinates
    shifted_points = [[(x - 90) / 90, -(y - 120) / 120] for (x, y) in palm_points]  # (90, 120) is the center in graph paper
    objp[:, :2] = shifted_points  # Left Hand

    # Load the intrinsic paremeters from the saved file from the calibration step.
    intrinsicCalibData = np.load("../data/CalibData.npz")
    intrinsicMatrix = intrinsicCalibData["mtx"]
    distCoeffs = intrinsicCalibData["dist"]

    with open("./objShaders/vertobj.vert", 'r') as f:  # V shader for background
        vertexObj = f.read()
    with open("./objShaders/fragobj.frag", 'r') as f:  # F shader for background
        fragmentObj = f.read()

    log.info("Loading Object!")
    local_dir = os.path.dirname(os.path.abspath(__file__))
    local_file = local_dir + "/" + filepath
    vertices, indices = data.get(local_file)
    brain = gloo.Program(vertexObj, fragmentObj)
    brain.bind(vertices)

    trackball = Trackball(Position("position"))
    brain['transform'] = trackball
    trackball.theta, trackball.phi, trackball.zoom = 0, 0, 40
    window.attach(brain['transform'])

    app.run()
    cv2.destroyAllWindows()
    cap.release()

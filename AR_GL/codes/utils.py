import numpy as np


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



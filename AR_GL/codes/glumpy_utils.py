import numpy as np
from glumpy import gloo


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


def update(trackball, glObj):
    model = trackball['model'].reshape(4, 4)
    view = trackball['view'].reshape(4, 4)
    projection = trackball['projection'].reshape(4, 4)
    glObj['m_model'] = model
    glObj['m_view'] = view
    glObj['m_projection'] = projection
    glObj['m_normal'] = np.array(np.matrix(np.dot(view, model)).I.T)


class Cube:
    def __init__(self, vertex, fragment):
        # Data for Cube
        self.V = np.zeros(8, [("position", np.float32, 3),
                              ("color", np.float32, 4)])
        self.V["position"] = np.array([[-0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0], [0.1, -0.1, 0],
                                       [0.1, -0.1, 0.2], [-0.1, -0.1, 0.2], [-0.1, 0.1, 0.2], [0.1, 0.1, 0.2]], dtype=np.float32)
        self.V["color"] = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
                           [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]
        self.V = self.V.view(gloo.VertexBuffer)

        self.I = np.array([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 1,
                           1, 6, 7, 1, 7, 2, 7, 4, 3, 7, 3, 2, 4, 7, 6, 4, 6, 5], dtype=np.uint32)
        self.I = self.I.view(gloo.IndexBuffer)
        self.cube = gloo.Program(vertex, fragment)
        self.cube.bind(self.V)


 # Cube object initialisation in main code
 #    cubeObj = Cube(vertex, fragment)


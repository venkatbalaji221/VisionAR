# /*************************************************/
# Author: A K Dash
# Description:
#           - Showing two windows one from OpenGL and other from OpenCV
#           - The textured cube on OpenGL window is dependent on the aruco marker from the opencv window
# Environment:
#           py2cv3 on ubuntu
# Dependency:
#           - PyGLM (Version: 0.4.8b1)
#
# ToDo
#       - Need to improve the rendering accuracy.
#       - Check the detailed theory behind projection matrix and if needed, chage the code accordingly
# /*************************************************/

from __future__ import division
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import numpy as np
import cv2
import argparse
import glm
import math
from pyoglar.preprocessing import ShaderLoader, TextureLoader
from pyoglar.processhand import Hand
from pyoglar.draw import draw3DAxis, draw3DCube

# GLFW window parameters
wTitle = "GLFW OpenGL window"
wWidth = 640
wHeight = 480


def key_callback(window, key, scancode, action, mode):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        print("Window is closing as the user has pressed the Esc key..")
        glfw.set_window_should_close(window, True)


def getImgp(i, hand_landmarks, img_wd, img_ht):
    x = int(hand_landmarks.landmark[i].x * img_wd)
    y = int(hand_landmarks.landmark[i].y * img_ht)
    mcp_point = [x, y]
    return mcp_point


def main():
    # Parse the command line arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("-t1", "--teximage1",
                    help="Path to the texture image1", required=True)
    ap.add_argument("-t2", "--teximage2",
                    help="Path to the Texture image2", required=True)
    args = vars(ap.parse_args())

    # initialize the glfw lib
    if not glfw.init():
        raise Exception("Unable to initialize glfw..")

    # setting the window hint parameters
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.RESIZABLE, False)

    # Creating the window.
    window = glfw.create_window(wWidth, wHeight, wTitle, None, None)

    # Check if windows was created successfully
    if not window:
        glfw.terminate()
        raise Exception("glfw window cannot be created..")

    width, height = glfw.get_framebuffer_size(window)
    print("Frame buffer width: {}, height: {}".format(width, height))

    # set the windows position in the screen
    glfw.set_window_pos(window, 800, 200)

    # make the context current
    glfw.make_context_current(window)

    # set the view port
    glViewport(0, 0, width, height)

    # set the Esc key callback function to close the window
    glfw.set_key_callback(window, key_callback)

    vertices = [
        # front face: (vertex) (texture coord)
        [-0.5, -0.5, 0.5, 0, 0],  # index 0
        [0.5, -0.5, 0.5, 1, 0],  # index 1
        [0.5, 0.5, 0.5, 1, 1],  # index 2
        [-0.5, 0.5, 0.5, 0, 1],  # index 3
        # Back face: (vertex) (texture coord)
        [-0.5, -0.5, -0.5, 0, 0],  # index 4
        [0.5, -0.5, -0.5, 1, 0],  # index 5
        [0.5, 0.5, -0.5, 1, 1],  # index 6
        [-0.5, 0.5, -0.5, 0, 1],  # index 7
        # left face: (vertex) (texture coord)
        [-0.5, -0.5, 0.5, 0, 0],  # index 8
        [-0.5, -0.5, -0.5, 1, 0],  # index 9
        [-0.5, 0.5, -0.5, 1, 1],  # index 10
        [-0.5, 0.5, 0.5, 0, 1],  # index 11
        # right face: (vertex) (texture coord)
        [0.5, -0.5, 0.5, 0, 0],  # index 12
        [0.5, -0.5, -0.5, 1, 0],  # index 13
        [0.5, 0.5, -0.5, 1, 1],  # index 14
        [0.5, 0.5, 0.5, 0, 1],  # index 15
        # top face: (vertex) (texture coord)
        [-0.5, 0.5, 0.5, 0, 0],  # index 16
        [0.5, 0.5, 0.5, 1, 0],  # index 17
        [0.5, 0.5, -0.5, 1, 1],  # index 18
        [-0.5, 0.5, -0.5, 0, 1],  # index 19
        # bottom face: (vertex) (texture coord)
        [-0.5, -0.5, 0.5, 0, 0],  # index 20
        [0.5, -0.5, 0.5, 1, 0],  # index 21
        [0.5, -0.5, -0.5, 1, 1],  # index 22
        [-0.5, -0.5, -0.5, 0, 1],  # index 23
    ]
    # convert the vertices list into a numpy array
    vertices = np.array(vertices, dtype=np.float32)
    indices = [0, 1, 2, 2, 3, 0,
               4, 5, 6, 6, 7, 4,
               8, 9, 10, 10, 11, 8,
               12, 13, 14, 14, 15, 12,
               16, 17, 18, 18, 19, 16,
               20, 21, 22, 22, 23, 20
               ]
    indices = np.array(indices, dtype=np.uint32)

    # / **********************BUFFERS**********************************/
    # Generate the buffers
    vao = glGenVertexArrays(1)
    # Bind the vertex array objects first before the vbo
    glBindVertexArray(vao)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Generate the vertex buffer object
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)  # put the data into buffer
    # send the data into VRAM
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))
    glBindVertexArray(0)  # Unbind the vao
    # / **********************BUFFERS END**********************************/
    # Load the Textures
    texobj = TextureLoader(2)
    texobj.loadtexture(args["teximage1"], args["teximage2"])

    # Build the shader program
    shaderobj = ShaderLoader("./shaders/vert_shader.vert",
                             "./shaders/frag_shader.frag")
    shaderProgram = shaderobj.processshaders()

    # find the location of uniform variable in the shader
    model_location = glGetUniformLocation(shaderProgram, "model")
    view_location = glGetUniformLocation(shaderProgram, "view")
    proj_persp_location = glGetUniformLocation(shaderProgram, "proj_persp")

    # [OpenCV] Setting up the parameters of the camera
    # markerObj = Marker('./config/MyWebcamCamParam.yml')
    # markerObj = Marker('./config/MyWebcamCalibData.npz')
    # markerObj.init_markerparams()
    handObj = Hand('./config/MyWebcamCalibData.npz')
    handObj.init_camparams()
    camMatrix = handObj.CameraMatrix
    distCoeff = handObj.Distorsion

    # Hand Obejct points from the hand calibration
    objp = np.zeros((6, 3), dtype=np.float32)
    # objp[:, :2] = [[69, 118], [92, 119], [118, 122], [144, 134]]  # Right hand coordinates
    objp[:, :2] = [[68, 214], [104, 198], [109, 124], [
        85, 118], [62, 124], [40, 137]]  # Left Hand

    # Sacling the matrix for the 3D object
    # half_size = glm.mat4(
    #     [0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1])
    #                       OR
    model_mat = glm.mat4(1)  # create the 4x4 identity matrix
    model_mat = glm.scale(model_mat, glm.vec3(0.085, 0.085, 0.085))
    model_mat = glm.translate(model_mat, glm.vec3(0, 0.0, 0.5))

    # default values of modelview so that initially when there is no marker,
    # the object is placed behind the camera and it does not show up
    view_mat = glm.mat4(1.0)
    # view_mat = glm.translate(view_mat, glm.vec3(0.0, 0.0, 10.0))

    # defining the projection matrix
    near = 0.1
    far = 500.0
    # change the type np.float32 to float as glm does not recognize the former type
    fx = np.float64(camMatrix[0][0])
    fy = np.float64(camMatrix[1][1])
    cx = np.float64(camMatrix[0][2])
    cy = np.float64(camMatrix[1][2])
    proj_persp_mat = glm.mat4(1.0)  # initialize projection persp matrix
    proj_persp_mat = glm.mat4([fx / cx, 0, 0, 0, 0, fy / cy, 0, 0, 0, 0, -(
            far + near) / (far - near), -(2 * far * near) / (far - near), 0, 0, -1, 0])
    # proj_persp_mat = glm.mat4(
    #     [2*fx/640, 0, (640-2*cx)/640, 0,
    #      0, -2*fy/480, (480 - 2*cy) / 480, 0,
    #      0, 0, -(far + near)/(far-near), -2.0*near*far/(far - near),
    #      0, 0, -1, 0
    #      ]
    # )
    proj_persp_mat = glm.transpose(proj_persp_mat)

    # Background object definitions
    vertices_bg = [
        [-0.5, -0.5, 0, 0, 0],
        [0.5, -0.5, 0, 1, 0],
        [0.5, 0.5, 0, 1, 1],
        [-0.5, 0.5, 0, 0, 1]
    ]
    vertices_bg = np.array(vertices_bg, dtype=np.float32)
    indices_bg = [0, 1, 2, 2, 3, 0]
    indices_bg = np.array(indices_bg, dtype=np.uint32)

    # / **********************Background BUFFERS**********************************/
    # Generate the buffers
    vao_bg = glGenVertexArrays(1)
    # Bind the vertex array objects first before the vbo
    glBindVertexArray(vao_bg)

    ebo_bg = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_bg)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_bg.nbytes, indices_bg, GL_STATIC_DRAW)

    # Generate the vertex buffer object
    vbo_bg = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_bg)
    glBufferData(GL_ARRAY_BUFFER, vertices_bg.nbytes, vertices_bg, GL_STATIC_DRAW)  # put the data into buffer
    # send the data into VRAM
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))
    glBindVertexArray(0)  # Unbind the vao_bg
    # / **********************BUFFERS END**********************************/
    # [OpenCV] Read the camera frames
    cap = cv2.VideoCapture(0)
    # build the background shader program
    shaderObj_bg = ShaderLoader(
        "./shaders/bg_vert_shader.vert", "./shaders/bg_frag_shader.frag")
    shaderProgram_bg = shaderObj_bg.processshaders()
    # ********************************************************
    # transformation presets for the background object (model, view and orthographic projection)
    modelview_bg = glm.mat4(1)
    modelview_bg = glm.mat4([1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, -499.0, 0, 0, 0, 1.0])
    modelview_bg = glm.transpose(modelview_bg)
    modelview_bg = glm.scale(modelview_bg, glm.vec3(998 * cx / fx, 998 * cy / fy, 0))

    # Loop until the user closed the window
    while not glfw.window_should_close(window):

        # poll and process the events
        glfw.poll_events()
        # render the contents here
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.5, 0.5, 1.0)

        # [OpenCV] Capture live camera feed
        ret, frame = cap.read()
        cv2.imshow('Live Video: Original feed', frame)
        # generate webcam texture buffer
        texobj_bg = TextureLoader(1)
        texobj_bg.loadtexture(frame)

        handObj.handDetector()  # Activate the hand detector
        handObj.handLandmarkDetect(frame)

        # markers = markerObj.markerDetect(frame)
        # glm stores the result in RMO. so, tx, ty and tz is at the last row
        # view_mat = glm.translate(view_mat, glm.vec3(0.0, 0.0, 10.0))
        # (OR)
        view_mat = glm.mat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 10, 1])

        if handObj.hand_detect_flag is True:
            for handlandmark in handObj.handlandmarks:
                # Not Righthandedness
                if not handlandmark.landmark[2].x < handlandmark.landmark[17].x:
                    imgp = np.zeros((6, 2), dtype=np.float32)
                    k = 0
                    for i in [0, 1, 5, 9, 13, 17]:
                        # in the below function call image_width = 640, image_height = 480
                        result_point = getImgp(i, handlandmark, 640, 480)
                        imgp[k, :] = np.array(result_point, dtype=np.float32)
                        k = k + 1

                    print("ObjectPoints:\n", objp)
                    print("ImagePoints:\n", imgp)

                    retval, rvec, tvec = cv2.solvePnP(objp, imgp, camMatrix, distCoeff)
                    # [OpenCV] Draw the cube and axis on opencv window
                    draw3DAxis(frame, camMatrix, distCoeff, rvec, tvec, 100, True)
                    # [OpenGL] for the 3d object on opengl window
                    rotMat, jacobian = cv2.Rodrigues(rvec)
                    offset = glm.vec3()
                    view_mat = glm.mat4([
                        np.float64(rotMat[0][0]), np.float64(rotMat[0][1]), np.float64(rotMat[0][2]), np.float64(tvec[0]),
                        -np.float64(rotMat[1][0]), -np.float64(rotMat[1][1]), -np.float64(rotMat[1][2]), -np.float64(tvec[1]),
                        -np.float64(rotMat[2][0]), -np.float64(rotMat[2][1]), -np.float64(rotMat[2][2]), -np.float64(tvec[2]),
                        0, 0, 0, 1.0]
                    )
                    # view_mat = glm.transpose(view_mat)

        glUseProgram(shaderProgram)
        # Setting the first texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texobj.textures[0])
        glUniform1i(glGetUniformLocation(shaderProgram, "crate_texture"), 0)

        # Setting the Second texture
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, texobj.textures[1])
        glUniform1i(glGetUniformLocation(shaderProgram, "baboon_texture"), 1)

        glUniformMatrix4fv(model_location, 1, GL_FALSE, glm.value_ptr(model_mat))
        glUniformMatrix4fv(view_location, 1, GL_FALSE, glm.value_ptr(view_mat))
        glUniformMatrix4fv(proj_persp_location, 1, GL_FALSE, glm.value_ptr(proj_persp_mat))

        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)

        # Draw background
        glUseProgram(shaderProgram_bg)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, texobj_bg.texture)
        glUniform1i(glGetUniformLocation(shaderProgram_bg, "webcam_texture"), 2)
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram_bg, "modelview_bg"), 1, GL_FALSE, glm.value_ptr(modelview_bg))
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram_bg, "proj_persp_bg"), 1, GL_FALSE, glm.value_ptr(proj_persp_mat))

        glBindVertexArray(vao_bg)
        glDrawElements(GL_TRIANGLES, len(indices_bg), GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glBindVertexArray(0)

        # Display the resulting frame
        cv2.imshow('Live Video: Cube overlayed', frame)
        k = cv2.waitKey(1)
        if k == 27 or k == ord('q') or k == ord('Q'):
            break

        glEnable(GL_DEPTH_TEST)
        # swap the front and back buffer
        glfw.swap_buffers(window)

    # Terminate glfw, free up the allocated resources
    glfw.terminate()


if __name__ == "__main__":
    main()

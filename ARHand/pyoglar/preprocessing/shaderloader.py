
# /**************************************/
#   Written by: A K Dash
#   checked Env:  Py2Cv3
#
# ToDo
#      - for the time being no more updates
# /************************************/
from __future__ import division
import cv2
import numpy as np
from OpenGL.GL import *
# from OpenGL.GL.shaders import compileProgram, compileShader


class ShaderLoader:
    '''
        (<path to vertexshader file>, <path to fragmentshader file>)
    '''

    def __init__(self, vspath, fspath):
        self.vspath = vspath
        self.fspath = fspath

    def readshaders(self):
        '''
        @params: void
        @output:
                vsrc, fsrc
        '''
        # read the vertex shader
        file = open(self.vspath)
        vsrc = file.read()
        file.close()

        # Read the fragment shader
        file = open(self.fspath)
        fsrc = file.read()
        file.close()
        return vsrc, fsrc

    def processshaders(self):
        '''
        @params: 
            void

        @output:
            shaderProgram
        '''
        vsrc, fsrc = self.readshaders()
        # The two lines below will be the default methods provided by OpenGl to compile the shader.
        # If you wish, you coould use them after uncommenting the import statement. However the error logs are not well documented.

        # shaderprogram = compileProgram(compileShader(
        #     vsrc, GL_VERTEX_SHADER), compileShader(fsrc, GL_FRAGMENT_SHADER))

        # Compile the vertex shader
        vertexShaderID = glCreateShader(GL_VERTEX_SHADER)
        # Mark the code line below. The python and c++ prototype differs a bit
        glShaderSource(vertexShaderID, vsrc)
        glCompileShader(vertexShaderID)
        # Debug the vertex Shader
        status = glGetShaderiv(vertexShaderID, GL_COMPILE_STATUS)
        if status == GL_TRUE:
            print("[INFO] Vertex shader compiled successfully.")
        else:
            raise RuntimeError("[ERROR] Vertex shader compilation failed \n{}".format(
                glGetShaderInfoLog(vertexShaderID)))

        # Compile the fragment shader
        fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER)
        # Mark the line below. The python and c++ prototype differs a bit
        glShaderSource(fragmentShaderID, fsrc)
        glCompileShader(fragmentShaderID)
        # Debug the vertex Shader
        status = glGetShaderiv(fragmentShaderID, GL_COMPILE_STATUS)
        if status == GL_TRUE:
            print("[INFO] Fragment shader compiled successfully.")
        else:
            raise RuntimeError("[ERROR] Fragment shader compilation failed: \n{}".format(
                glGetShaderInfoLog(fragmentShaderID)))

        # Attaching the shader into program
        shaderProgram = glCreateProgram()
        glAttachShader(shaderProgram, vertexShaderID)
        glAttachShader(shaderProgram, fragmentShaderID)
        # link the shader program
        glLinkProgram(shaderProgram)

        # check for any linking errors.
        status = glGetProgramiv(shaderProgram, GL_LINK_STATUS)
        if status == GL_TRUE:
            print("[INFO] Shader linked successfully.")
        else:
            raise RuntimeError("[ERROR] Shader failed to link. \n{}".format(
                glGetProgramInfoLog(shaderProgram)))

        return shaderProgram

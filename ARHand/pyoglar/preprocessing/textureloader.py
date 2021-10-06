# /**************************************/
#   Written by: A K Dash
#   checked Env:  Py2Cv3
#
# ToDo
#      - generalize it for images and for ndarray
#      - works with as many textures as you want.( tested with 2 textures)
# /************************************/
from __future__ import division
from cv2 import imread, flip
from OpenGL.GL import glBindTexture, glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, \
    GL_TEXTURE_WRAP_T, GL_REPEAT, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,\
    glTexImage2D, GL_RGBA, GL_UNSIGNED_BYTE, glGenTextures, glGenerateMipmap, GL_RGB, GL_BGR


class TextureLoader:
    '''
        <n> number of texture id to generate
    '''

    def __init__(self, n=1):
        if n == 1:
            self.texture = glGenTextures(1)
        else:
            self.textures = glGenTextures(n)

    def loadtexture(self, *args):
        '''
        @params:
        *args
            provides the <n> paths to the texture images e.g. ( imagepath_1, imagepath_2, ... imagepath_(n-1) )

        @output:
            No return val. Inplace update of
            self.textures[0], self.textures[1], ... self.textures[n-1]

        '''

        for (i, filename) in enumerate(args):
            if type(filename) == str:
                image = imread(filename)
                # Flip the image vertically to coincide the input image with the OpenGl texture coordinate
                imageFlip_y = flip(image, 0)
                glBindTexture(GL_TEXTURE_2D, self.textures[i])
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                             image.shape[1], image.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, imageFlip_y)
                # Set the texture wrapping parameters
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                # Set texture filtering parameters
                glTexParameteri(
                    GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(
                    GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glGenerateMipmap(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, 0)
            else:
                # Flip the image vertically to coincide the input image with the OpenGl texture coordinate
                imageFlip_y = flip(filename, 0)
                glBindTexture(GL_TEXTURE_2D, self.texture)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                             filename.shape[1], filename.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, imageFlip_y)
                # Set the texture wrapping parameters
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                # Set texture filtering parameters
                glTexParameteri(
                    GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(
                    GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glGenerateMipmap(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, 0)

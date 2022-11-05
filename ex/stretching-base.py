from typing import List
from mpi4py import MPI
import sys
import random
import math
import numpy as np
import matplotlib.image as mpimg
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy

M = 255

# First method for stretching contrast


def f_one(x, n):
    if x == 0:
        return 0
    return int(M**(1 - n) * (x**n))

# Second method for stretching contrast


def f_two(x, n):
    if x == 0:
        return 0
    return int((M**((n - 1) / n)) * (x ** (1 / n)))

# Converts an image to grayscale


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

# splits a vector "x" in "size" part. In case it does not divide well, the last one receives one less than others


def split(x, size):
    n = math.ceil(len(x) / size)
    return [x[n * i:n * (i + 1)] for i in range(size - 1)] + [x[n * (size - 1):len(x)]]

# unsplits a list x composed of n lists of t elements


def unsplit(x):
    y = []
    n = len(x)
    t = len(x[0])
    for i in range(n):
        for j in range(len(x[i])):
            y.append(x[i][j])
    return y, n, t

# Loads an image on disk named "image.png" and convert it to greyscale, and shows it


def readImage():
    img = mpimg.imread('image.png')
    # print(img.shape)
    plt.imshow(img)
    plt.show()
    grey = rgb2gray(img)
    plt.imshow(grey, cmap=cm.Greys_r)
    pixels, nblines, nbcolumns = unsplit(grey)
    for i in range(0, len(pixels)):
        pixels[i] = int(pixels[i] * 255)
    return pixels, nblines, nbcolumns

# Saves the image in "image-grey2-stretched.png" and shows it


def saveImage(newP, nblines, nbcolumns):
    newi = split(newP, nblines)
    newimg = np.zeros((nblines, nbcolumns))
    for rownum in range(nblines):
        for colnum in range(nbcolumns):
            newimg[rownum][colnum] = newi[rownum][colnum]
    plt.imshow(newimg, cmap=cm.Greys_r)
    plt.show()
    mpimg.imsave('image-grey2-stretched.png', newimg, cmap=cm.Greys_r)


print("Starting stretching...")

comm = MPI.COMM_WORLD
rank: int = comm.rank
size: int = comm.size

if rank == 0:
    # load the image
    pixels, nblines, nbcolumns = readImage()
    split_pixels = split(pixels, size)
else:
    split_pixels = None

s_pixels = comm.scatter(split_pixels, root=0)
# compute min and max of pixels
l_pix_min = min(s_pixels)
l_pix_max = max(s_pixels)
pix_min = comm.reduce(l_pix_min, op=MPI.MIN, root=0)
pix_max = comm.reduce(l_pix_max, op=MPI.MAX, root=0)
if rank == 0:
    # compute alpha, the parameter for f_* functions
    alpha = 1 + (pix_max - pix_min) / M
else:
    alpha = None

alpha = comm.bcast(alpha, root=0)

# stretch contrast for all pixels. f_one and f_two are the two different method
for i in range(len(s_pixels)):
    if rank % 2 == 0:
        s_pixels[i] = f_one(s_pixels[i], alpha)
    else:
        s_pixels[i] = f_two(s_pixels[i], alpha)

g_pixels: list = comm.gather(s_pixels, root=0)

if rank == 0:
    pixels = unsplit(g_pixels)[0]

    # save the image
    saveImage(pixels, nblines, nbcolumns)
    print("Stretching done...")

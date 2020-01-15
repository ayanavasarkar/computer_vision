# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji

import numpy as np
import matplotlib.pyplot as plt
from utils import imread
from depthFromStereo import depthFromStereo
import os

read_path = "../data/disparity/"
im_name1 = "tsukuba_im1.jpg" 
im_name2 = "tsukuba_im5.jpg"
#Read test images
# img1 = imread(os.path.join(read_path, im_name1))
# img2 = imread(os.path.join(read_path, im_name2))


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# imgL = cv.imread('tsukuba_l.png',0)
# imgR = cv.imread('tsukuba_r.png',0)
img1 = cv.imread("tsukuba_im1.jpg" , 0)
img2 = cv.imread("tsukuba_im5.jpg", 0 )
img1 = np.pad(img1, 100, mode='constant')
img2 = np.pad(img2, 100, mode='constant')

stereo = cv.StereoBM_create(numDisparities=160, blockSize=15)
disparity = stereo.compute(img1,img2)
plt.imshow(disparity,'gray')
plt.show()
exit()
#Compute depth
depth = depthFromStereo(img1, img2, 23)
# depth =10
#Show result
plt.imshow(depth, cmap='gray')
plt.show()
save_path = "../output/disparity/"
save_file = "tsukuba.png"
if not os.path.isdir(save_path):
	os.makedirs(save_path)
plt.imsave(os.path.join(save_path, save_file), depth)



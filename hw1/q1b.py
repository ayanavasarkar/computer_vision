import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import math, random

S = np.random.rand(1,3) # Light vector
x = random.randint(0,10000)
y = random.randint(0,10000)

r = 1
l_intensity = 1

c = [0,255,255 ]    # Random RGB value

theta = math.degrees(math.atan(y/x))
img = np.linalg.norm(S) * math.cos(theta)

# img = cv2.imread('blue.png', 0)

# im = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
# plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
# plt.show()

# c = cv2.addWeighted(img, 2.5, np.zeros(img.shape, img.dtype), 0, 0)
# ret, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
#
# img[thresh == 255] = 0
#
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# erosion = cv2.erode(img, kernel, iterations = 1)
# plt.imshow(img, cmap='gray')
# plt.show()
# exit(0)


# img = np.zeros((64,64,3), dtype=np.uint8)
img = cv2.imread("t1.png")
imgsize = img.shape[:2]
in_color = (255,255,255)
out_color = (0,0,0)

for y in range(imgsize[0]):
    for x in range(imgsize[1]):
        dist_center = np.sqrt((x) ** 2 + (y - imgsize[1]) ** 2)

        dist_center = (dist_center) / (np.sqrt(4) * imgsize[0]/2)

        r = out_color[0] * dist_center + in_color[0] * (1 - dist_center)
        g = out_color[1] * dist_center + in_color[1] * (1 - dist_center)
        b = out_color[2] * dist_center + in_color[2] * (1 - dist_center)
        # print r, g, b

        img[y, x] = (int(r), int(g), int(b))

arr1=img
arr2=img[::-1,:,:]
arr3=img[::-1,::-1,:]
arr4=img[::,::-1,:]
arr5=np.vstack([arr1,arr2])
arr6=np.vstack([arr4,arr3])
arr7=np.hstack([arr6,arr5])
plt.imshow(arr7, cmap='gray')
plt.show()
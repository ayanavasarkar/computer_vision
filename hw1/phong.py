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

##Difusse+Specular
img = np.linalg.norm(S) * math.cos(theta)

a = -0.5
b = -1
c = 250
width = 55

n = 4

# img = np.zeros((64,64,3), dtype=np.uint8)
img = cv2.imread("p1.png")
imgsize = img.shape[:2]
in_color = (255,255,255)
out_color = (0,0,0)
n=50
for y in range(imgsize[0]):
    for x in range(imgsize[1]):
        dist_center = np.sqrt((x) ** 2 + (y - imgsize[1]) ** 2)

        dist_center = (dist_center) / (np.sqrt(n) * imgsize[0]/2)

        r = out_color[0] * dist_center + in_color[0] * (1 - dist_center)
        g = out_color[1] * dist_center + in_color[1] * (1 - dist_center)
        b = out_color[2] * dist_center + in_color[2] * (1 - dist_center)
        # print r, g, b

        img[y, x] = (int(r), int(g), int(b))

innerColor = [0,0,0] #Color at the center
outerColor = [255,255,255] #Color at the edge

for y in range(imgsize[1]):
    for x in range(imgsize[0]):

        dist = (a*x + b*y + c)/np.sqrt(a*a+b*b)
        color_coef = abs(dist)/width

        if abs(dist) < width:
            red = outerColor[0] * color_coef + innerColor[0] * (1 - color_coef)
            green = outerColor[1] * color_coef + innerColor[1] * (1 - color_coef)
            blue = outerColor[2] * color_coef + innerColor[2] * (1 - color_coef)

            img[x, y]= (int(red), int(green), int(blue))

arr1=img
arr2=img[::-1,:,:]
arr3=img[::-1,::-1,:]
arr4=img[::,::-1,:]
arr5=np.vstack([arr1,arr2])
arr6=np.vstack([arr4,arr3])
arr7=np.hstack([arr6,arr5])
plt.imshow(arr7, cmap='gray')
plt.show()
##rendering of a sphere from the light source for a pure lambertian object
import numpy as np
import math, random
#
# S = np.random.rand(1,3) # Light vector
# x = random.randint(0,10000)
# y = random.randint(0,10000)
#
# r = 1
# l_intensity = 1
#
# c = [0,255,255 ]    # Random RGB value
#
# theta = math.degrees(math.atan(y/x))
# B = np.linalg.norm(S) * math.cos(theta)
#
# print(B, x, y)
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')
#
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
#
# x = 1 * np.outer(np.cos(u), np.sin(v))
# y = 1 * np.outer(np.sin(u), np.sin(v))
# z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
#
# elev = 10.0
# rot = 80.0 / 180 * np.pi

import cv2

img = cv2.imread("phong.png", 0)
cv2.imwrite("phong_bw.png", img)
# plt.show()
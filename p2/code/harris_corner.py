from skimage.feature import corner_harris, corner_peaks
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from scipy import signal as sig
import numpy as np
import cv2 as cv

import seaborn as sb

# img = imread("tsukuba_im1.jpg")
img = imread("lion.png")
imggray = rgb2gray(img)

# h, w = imggray.shape
# print(imggray.shape)
# kernel = np.ones((5,5), dtype = float)
# im = sig.convolve2d(imggray, kernel, mode='same')
# plt.axis('off')
# plt.imshow(im, cmap="gray")
# plt.show()
#
# exit(0)




def gradient_x(imggray):
    ##Sobel operator kernels.
    # kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    # kernel_x = np.array([[-1, 0, 1], [1, 0, 1], [-1, 0, 1]])    ##BEST PARA
    # kernel_x = np.array([[0,0,0], [0, 1, 0], [0, 0, 0]])
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])       ##gradient along x
    return sig.convolve2d(imggray, kernel_x, mode='same')
def gradient_y(imggray):
    # kernel_y = np.array([[-1, 0, 1], [0, 0, 0], [-1, 0, 1]])
    # kernel_y = np.array([[1, 0, 1], [0, 0, 0], [-1, 0, -1]])    ##BEST PARA
    # kernel_y = np.array([[0,0,0], [0, 1, 0], [0, 0, 0]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])   ##gradient along y
    return sig.convolve2d(imggray, kernel_y, mode='same')


I_x = gradient_x(imggray)
I_y = gradient_y(imggray)
plt.axis('off')
plt.imshow(I_x, cmap="gray")
plt.show()
plt.axis('off')
plt.imshow(I_y, cmap="gray")
plt.show()
exit(0)



laplacian = cv.Laplacian(imggray,cv.CV_64F)
I_x = cv.Sobel(imggray,cv.CV_64F,1,0,ksize=3)
I_y = cv.Sobel(imggray,cv.CV_64F,0,1,ksize=3)


Ixx = I_x**2
Ixy = I_y*I_x
Iyy = I_y**2

k = 0.03

height, width = imggray.shape
harris_response = []
window_size = 10
offset = 1
R = np.zeros((height, width), dtype = float)
for y in range(offset, height - offset):
    for x in range(offset, width - offset):
        Sxx = np.sum(Ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
        Syy = np.sum(Iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
        Sxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

        # Find determinant and trace, use to get corner response
        det = (np.absolute((Sxx * Syy) - (Sxy ** 2)))
        trace = Sxx + Syy
        r = np.absolute(det - k * (trace ** 2))

        harris_response.append([x, y, r])
        R[y,x] = float(r)

from matplotlib.colors import ListedColormap

# print(R, np.max(R), np.min(R), np.mean(R))
cmap = sb.cubehelix_palette(light=1, as_cmap=True)
pal = sb.dark_palette("palegreen", as_cmap=True)

midpoint = (np.max(R) - np.min(R)) / 2

heat_map = sb.heatmap(R, center=midpoint, vmin=np.min(R), vmax=np.max(R))
plt.show()
exit(0)
img_copy = np.zeros((img.shape[0], img.shape[1]), dtype = float)
# print(r, r.shape, np.mean(r))
for response in harris_response:
    x, y, r = response
    if r > 0:
        img_copy[y, x] = 1 #[255, 0, 0]

plt.imshow(img_copy, cmap="gray")
plt.show()

# coords = corner_peaks(corner_harris(imggray))
#
# fig, ax = plt.subplots()
# ax.imshow(coords, interpolation='nearest', cmap=plt.cm.gray)
# ax.plot(coords[:, 1], coords[:, 0], '.r', markersize=3)
# plt.show()

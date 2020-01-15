import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('lion.png')
print(np.max(img), np.min(img), img.shape, img[0,0])

# img = np.delete(img, (0), axis = 0)
# img = np.delete(img, (0), axis = 1)
# img = np.delete(img, (img.shape[1]-1), axis = 1)
# img = np.delete(img, (img.shape[0]-1), axis = 0)

x = np.zeros((img.shape[0], img.shape[1]-1))
y = np.zeros((img.shape[0], img.shape[1]-1))

x = img[:, 0:img.shape[1]-2]
y = img[:, 1:img.shape[1]-1]

print(np.max(img), x, y, img[0,0])
# print(img[:,0:img.shape[1]-1]==x)

plt.scatter(y, x)
plt.show()

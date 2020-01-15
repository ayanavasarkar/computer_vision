import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def depthFromStereo(img1, img2, ws):
    img1 = np.dot(img1[...,:3], [0.2989, 0.5870, 0.1140])
    img2 = np.dot(img2[...,:3], [0.2989, 0.5870, 0.1140])

    depth_map = np.ones((img1.shape[0], img1.shape[1]))

    img1 = np.pad(img1, ws, mode='constant')
    img2 = np.pad(img2, ws, mode='constant')
    # print(img1.shape, img2.shape)
    disparity_mat = np.zeros((img1.shape[0], img1.shape[1]))

    for i in range(ws,img1.shape[0]):
        for j in range(ws, img2.shape[1]):
            min_ssd = 100000
            for k in range(ws, img2.shape[1]):

                try:
                    # ssd = np.sum((np.subtract(img1[i-ws:i+ws,j-ws:j+ws], img2[i-ws:i+ws,k-ws:k+ws]))**2)
                    ssd = cdist(img1[i-ws:i+ws, j-ws:j+ws], img2[i-ws:i+ws, k-ws:k+ws], 'sqeuclidean')
                    if(ssd < min_ssd):
                        min_ssd = ssd
                        x = np.absolute(k-j)
                except:
                    break

            disparity_mat[i - ws, j - ws] = x

        print(i, j)

    return np.divide(1, disparity_mat)

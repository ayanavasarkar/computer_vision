import numpy as np
import cv2
import matplotlib.pyplot as plt

def alignChannels(img, max_shift):

    # b, g, r = cv2.split()
    b = (img[:, :, 0])#.flatten()
    g = (img[:, :, 1])#.flatten()
    r = (img[:, :, 2])#.flatten()

    # b_e = cv2.Canny(np.uint8(b), 0, 1)
    # g_e = cv2.Canny(np.uint8(g), 0, 1)
    # r_e = cv2.Canny(np.uint8(r), 0, 1)

    g_min = 100000000
    r_min = 100000000

    index = np.zeros((2, 2), dtype=int)

    for i in range(-max_shift[0], max_shift[0], 1):
        for j in range(-max_shift[1], max_shift[1], 1):

            g_t = (np.roll(g, [i, j],  axis=[0, 1]))
            r_t = (np.roll(r, [i, j],  axis=[0, 1]))

            # print(sum(sum(np.square(np.subtract(g, b_t)))), np.mean(np.subtract(g, b_t)))
            # exit(0)
            g_ = float(sum(sum(np.square(np.subtract(b, g_t)))))
            r_ = float(sum(sum(np.square(np.subtract(b, r_t)))))


            if(g_ <= float(g_min)):
                g_min = float(g_)
                index[0, 0] = i
                index[0, 1] = j

            if(r_ <= float(r_min)):
                r_min = float(r_)
                index[1, 0] = i
                index[1, 1] = j

    g = np.roll(g, [index[0, 0], index[0, 1]],  axis=[0, 1])
    r = np.roll(r, [index[1, 0], index[1, 1]],  axis=[0, 1])

    rgbArray = np.dstack((b,g,r))
    try:
        rgbArray = rgbArray[30:rgbArray.shape[0]-50, 30:rgbArray.shape[1]-70]
    except:
        pass
    return (rgbArray, index)


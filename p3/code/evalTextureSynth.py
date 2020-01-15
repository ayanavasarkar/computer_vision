import numpy as np
from skimage import io
import cv2 as cv
from matplotlib import pyplot as plt


# Load images
# img = cv.imread('../data/texture/D20.png', 0)
img = cv.imread('../data/texture/Texture2.bmp', 0)
# img = cv.imread('../data/texture/english.jpg', 0)
# print(img.shape, img)
new_img = np.zeros((70, 70), dtype = float)

# plt.axis('off')
# plt.imshow(img, cmap="gray")
# plt.show()
# exit(0)
i=0
row =0
col =0
tile =5
while(i < 14*14):
    r = np.random.randint(0, img.shape[0]-tile)
    c = np.random.randint(0, img.shape[1]-tile)
    # print(r, c)
    patch = img[r:r+tile, c:c+tile]
    print(row, col, "##")
    new_img[row:row+tile, col:col+tile] = patch
    # plt.axis('off')
    # plt.imshow(patch, cmap="gray")
    # plt.show()
    # print(patch)
    # exit(0)
    if(col >= 65):
        row += tile
        col = 0
        print(col)
    else:
        col += tile
    i+=1


plt.axis('off')
plt.imshow(new_img, cmap="gray")
plt.show()
exit(0)

# Random patches
tileSize = 30 # specify block sizes
numTiles = 5
outSize = numTiles * tileSize # calculate output image size
# implement the following, save the random-patch output and record run-times
im_patch = synthRandomPatch(im, tileSize, numTiles, outSize)


# Non-parametric Texture Synthesis using Efros & Leung algorithm  
winsize = 11 # specify window size (5, 7, 11, 15)
outSize = 70 # specify size of the output image to be synthesized (square for simplicity)
# implement the following, save the synthesized image and record the run-times
im_synth = synthEfrosLeung(im, winsize, outSize)

iMAGE 1
3 to 15
('Total Time: ', 158.13485622406006, ' seconds')
('Total Time: ', 51.245678186416626, ' seconds')
('Total Time: ', 45.44844603538513, ' seconds')
('Total Time: ', 37.33331799507141, ' seconds')
('Total Time: ', 32.3615140914917, ' seconds')
('Total Time: ', 16.83189296722412, ' seconds')


Image 2
15 to 3
('Total Time: ', 25.65518593788147, ' seconds')
('Total Time: ', 36.941904067993164, ' seconds')
('Total Time: ', 51.220242977142334, ' seconds')
('Total Time: ', 65.95525693893433, ' seconds')
('Total Time: ', 66.05535197257996, ' seconds')

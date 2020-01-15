import numpy as np
from scipy import ndimage
from skimage.transform import resize
import matplotlib.pyplot as plt

# EXTRACTDIGITFEATURES extracts features from digit images
#   features = extractDigitFeatures(x, featureType) extracts FEATURES from images
#   images X of the provided FEATURETYPE. The images are assumed to the of
#   size [W H 1 N] where the first two dimensions are the width and height.
#   The output is of size [D N] where D is the size of each feature and N
#   is the number of images.

#  cols X rows X n_images
def extractDigitFeatures(x, featureType):
    
    if featureType == 'pixel':
        features = pixel_feat(x)
    elif featureType == 'hog':
        features = HOG(x)
    elif featureType == 'lbp':
        features = LBP(x)

    print(type(features), features.shape)
    return features


def SqrtNorm(data):
    return(np.sqrt(data))

def L2Norm(data):
    return (data / np.linalg.norm(data))

def pixel_feat(x):

    a = np.reshape(x, (28*28, x.shape[2]))

    for i in range(a.shape[1]):
        a[:, i] = L2Norm(a[:, i])

    return(a)

def HOG(x):

    # dat = np.transpose(np.reshape(x, (28*28, x.shape[2])))
    # data = np.reshape(dat, (x.shape[2], 28, 28))
    # return(compute_gradients(data, 10, 7))

    num_ori = 9
    num_bins = 7

    features = np.zeros((num_bins * num_bins * num_ori, x.shape[2]))
    cell_x = x.shape[0] / num_bins
    cell_y = x.shape[1] / num_bins
    # print(cell_x, cell_y, features.shape)
    for index in range(x.shape[2]):
        hog_feature = np.zeros((num_bins, num_bins, num_ori))
        image = x[:, :, index]
        dx, dy = np.gradient(image)
        curr_orientation = np.arctan2(dx, dy)
        curr_magnitude = np.sqrt(np.square(dx) + np.square(dy))
        for i in range(num_bins):
            for j in range(num_bins):
                ori_block = curr_orientation[i * cell_x:(i + 1) * cell_x + 1, j * cell_y:(j + 1) * cell_y + 1]
                mag_block = curr_magnitude[i * cell_x:(i + 1) * cell_x + 1,
                            j * cell_y:(j + 1) * cell_y + 1]

                for w in range(cell_x):
                    for h in range(cell_y):
                        ori = int(ori_block[w, h])


                        hog_feature[i, j, ori] += mag_block[w, h]

        feature = hog_feature.ravel()
        feature = SqrtNorm(feature)
        features[:, index] = feature

    return features

def compute_gradients(im, bin_size, cell_size):
    ## take bins from 0-90 with bin_size =10

    hist = np.zeros((2000, 4 * 4, 9), dtype=float)
    for img_n in range(0, im.shape[0]):

        img = im[img_n]
        gx, gy = np.gradient(img)

        mag = np.sqrt(np.square(gx) + np.square(gy))
        direction = (np.arctan2(gy, gx))

        hist_count = 0
        for r in range(0, 28, 7):
            for c in range(0, 28, 7):
                mask = (direction[r:r+7, c:c+7].astype(int)) // 10
                his = mag[r:r+7, c:c+7]
                # ssd = np.sqrt((np.square(his)).sum())
                # his = his / ssd
                for i in range(0, mask.shape[0]):
                    for j in range(0, mask.shape[1]):
                        hist[img_n, hist_count, mask[i, j]] += (his[j, i])
                hist_count+=1
        # hist[img_n] = hist[img_n].ravel()
        hist[img_n] = L2Norm(hist[img_n])
        # features[:, index] = feature

    hist = np.transpose(np.reshape(hist, (hist.shape[0], hist.shape[1]*hist.shape[2])))
    hist = np.reshape(hist, (16 * 9, 2000))
    return(hist)

def LBP(x):

    lbp = np.zeros((256, x.shape[2]))

    neighbor = 3
    vec = np.array([[1,2, 4], [8,0,16], [32,64,128]])

    for i in range(x.shape[2]):
        im = x[:, :, i]

        for h in range(0, im.shape[0] - neighbor):
            for w in range(0, im.shape[1] - neighbor):
                img = im[h:h + neighbor, w:w + neighbor]
                center = img[1, 1]
                patch = np.subtract(img, center)
                patch[patch > 0] = 1
                patch[patch < 0] = 0
                n = int(np.sum(np.sum(np.multiply(patch, vec))))
                lbp[n, i] += 1
        # lbp[0, i] = 0
        lbp[:, i] = L2Norm(lbp[:, i])

    return (lbp)
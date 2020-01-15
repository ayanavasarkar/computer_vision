# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2018
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 1 

import numpy as np

def demosaicImage(image, method='nn'):
    ''' Demosaics image.

    Args:
        img: np.array of size NxM.
        method: demosaicing method (baseline or nn).

    Returns:
        Color image of size NxMx3 computed using method.
    '''
    # print(image.shape, (image[0]))

    if method.lower() == "baseline":
        return demosaicBaseline(image.copy())
    elif method.lower() == 'nn':
        return demosaicNN(image.copy()) # Implement this
    elif method.lower() == 'linear':
        return demosaicLinear(image.copy()) # Implement this
    elif method.lower() == 'adagrad':
        return demosaicAdagrad(image.copy()) # Implement this
    else:
        raise ValueError("method {} unkown.".format(method))


def demosaicBaseline(img):
    '''Baseline demosaicing.
    
    Replaces missing values with the mean of each color channel.
    
    Args:
        img: np.array of size NxM.

    Returns:
        Color image of sieze NxMx3 demosaiced using the baseline 
        algorithm.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    red_values = img[0:image_height:2, 0:image_width:2]
    mean_value = red_values.mean()
    mos_img[:, :, 0] = mean_value
    mos_img[0:image_height:2, 0:image_width:2, 0] = img[0:image_height:2, 0:image_width:2]

    blue_values = img[1:image_height:2, 1:image_width:2]
    mean_value = blue_values.mean()
    mos_img[:, :, 2] = mean_value
    mos_img[1:image_height:2, 1:image_width:2, 2] = img[1:image_height:2, 1:image_width:2]

    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1
    green_values = mos_img[mask > 0]
    mean_value = green_values.mean()

    green_channel = img
    green_channel[mask < 0] = mean_value
    mos_img[:, :, 1] = green_channel

    return mos_img


def demosaicNN(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''

    final_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = float)
    # print(final_img.shape)
    # img = np.expand_dims(img, axis=2)
    print(img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(i%2 == 0 and j%2==0):
                try:
                    final_img[i,j,1] = float((img[i,j+1] +img[i+1, j])/2)
                    final_img[i, j, 0] = img[i+1, j + 1]
                    final_img[i,j,2] = img[i,j]
                except:
                    final_img[i, j, 1] = float((img[i, j - 1] + img[i - 1, j]) / 2)
                    final_img[i, j, 0] = float((img[i - 1, j - 1]))
                    final_img[i, j, 2] = img[i, j]

            elif (i % 2 == 1 and j % 2 == 1):
                try:
                    final_img[i, j, 1] = float((img[i, j + 1] + img[i + 1, j]) / 2)
                    final_img[i, j, 2] = img[i + 1, j + 1]
                    final_img[i, j, 0] = img[i, j]
                except:
                    final_img[i, j, 1] = float((img[i, j - 1] + img[i - 1, j]) / 2)
                    final_img[i, j, 2] = float((img[i - 1, j - 1]))
                    final_img[i, j, 0] = img[i, j]
    print((final_img))
    exit(0)
    return 0


def demosaicLinear(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    return demosaicBaseline(img)


def demosaicAdagrad(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    return demosaicBaseline(img)

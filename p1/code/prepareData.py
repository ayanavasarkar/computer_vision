import numpy as np

def prepareData(imArray, ambientImage):

    imArray = np.transpose(imArray, (2, 0, 1))
    # print(imArray.shape[0])
    # exit()
    ambientImage = np.repeat(ambientImage[np.newaxis, :, :], imArray.shape[0], axis=0)
    imArray = np.subtract(imArray, ambientImage)
    imArray[imArray<0]=0
    xmax, xmin = imArray.max(), imArray.min()
    imArray = (imArray - xmin) / (xmax - xmin)
    # print(imArray, xmin, xmax)

    return(np.transpose(imArray,(1,2,0)))
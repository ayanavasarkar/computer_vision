import numpy as np
from scipy.linalg import lstsq

def photometricStereo(imarray, lightdirs):

    print(imarray.shape, lightdirs.shape)
    im = (imarray.reshape(imarray.shape[2], imarray.shape[0] * imarray.shape[1]))

    print(lightdirs.shape, im.shape)
    p, res, rnk, s = lstsq(lightdirs, im)

    p = p.reshape(3, 192, 168)
    print(p.shape, s)
    p1 = np.square(p)
    p1 = np.sum(p1, axis =0)
    albedo = np.sqrt(p1)

    print(p1.shape)
    a = np.repeat(albedo[np.newaxis, :, :], 3, axis=0)
    surfaceNormals = np.divide(p, albedo)
    exit(0)
    return(0, 0)
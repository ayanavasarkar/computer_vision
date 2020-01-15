import numpy as np

from multiclassLRTrain import multiclassLRTrain

def trainModel(x, y):
    param = {}
    param['lambda'] = 0.0001      # Regularization term
    param['maxiter'] = 10000    # Number of iterations
    param['eta'] = 0.01         # Learning rate

    return multiclassLRTrain(x, y, param)

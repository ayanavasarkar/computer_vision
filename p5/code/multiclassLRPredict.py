import numpy as np

def multiclassLRPredict(model, x):
    W = model['w'].T
    x = x.T

    f = x.dot(W)
    f = f - np.max(f, axis=1).reshape(-1, 1)

    exp_f = np.exp(f)
    p = exp_f / np.sum(exp_f, axis=1)[:, np.newaxis]
    ypred = np.ones(x.shape[0])
    for i in np.arange(0, p.shape[0]):
        ypred[i] = model['classLabels'][np.argmax(p[i, :])]
    return ypred

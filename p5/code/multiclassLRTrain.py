import numpy as np

def softmax_loss(W, bias, x, y, l):

    W = np.transpose(W)
    x = np.transpose(x)
    n_samples = x.shape[0]

    f = x.dot(W) + bias
    # print(np.max(f, axis=1).reshape(-1, 1).shape, f.shape, x.shape, W.shape)
    f -= np.max(f, axis=1).reshape(-1, 1)
    exp_f = np.exp(f)
    p = exp_f / np.sum(exp_f, axis=1)[:, np.newaxis]

    loss = np.sum(-np.log(p[np.arange(n_samples), y]))
    loss /= n_samples
    loss += l * np.sum(W * W)

    p[np.arange(n_samples), y] -= 1

    dW = p.T.dot(x)
    dW = dW / n_samples
    dW = dW.T
    dW = dW + (2 * l * W)
    dW = dW.T
    return loss, dW

def multiclassLRTrain(x, y, param):

    classLabels = np.unique(y)
    numClass = classLabels.shape[0]
    numFeats = x.shape[0]

    # Initialize weights randomly
    model = {}
    model['w'] = np.random.randn(numClass, numFeats)*0.01
    model['bias'] = np.ones(numClass)
    model['classLabels'] = classLabels

    #(Implement gradient descent)
    ## Batch Gradient Descent
    max_epoch = param['maxiter']
    eta = param['eta']
    l = param['lambda']

    for epoch in range(max_epoch):
        loss, grad = softmax_loss(model['w'], model['bias'], x, y, l)
        # if loss == np.inf:
        #     return model
        model['w'] = model['w'] - eta * grad

    return model

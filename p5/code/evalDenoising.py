# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#

import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from dip import EncDec
from utils import imread
import torch.nn.functional as F

# Load clean and noisy image
#im = imread('../data/denoising/saturn.png')
#noise1 = imread('../data/denoising/saturn-noise1g.png')
im = imread('../data/denoising/lena.png')
noise1 = imread('../data/denoising/lena-noisy.png')

error1 = ((im - noise1)**2).sum()

print('Noisy image SE: {:.2f}'.format(error1))

# plt.figure(1)
#
# plt.subplot(121)
# plt.imshow(im, cmap='gray')
# plt.title('Input')
#
# plt.subplot(122)
# plt.imshow(noise1, cmap='gray')
# plt.title('Noisy image SE {:.2f}'.format(error1))

# plt.show()

#Create network
model = EncDec()

# Loads noisy image and sets it to the appropriate shape
noisy_img = torch.FloatTensor(noise1).unsqueeze(0).unsqueeze(0).transpose(2, 3)
clean_img = torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).transpose(2,3)
# Creates \eta (noisy input)
eta = torch.randn(*noisy_img.size())
eta.detach()

batch_size = 128
learning_rate = 1e-3
max_epoch = 1000

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

print(noisy_img.shape, clean_img.shape)

def train(model, train_data, noisy_data, optimizer):
    # Squared error
    sse = F.mse_loss
    optimizer.zero_grad()
    output = model(train_data)
    a_loss = sse(output, noisy_data)
    loss = sse(output, noisy_data, reduction='sum')
    loss.backward()
    optimizer.step()
    # print("Training Loss :%.4f Standard Error:%.4f"%(a_loss, loss))
    return loss, output


def test(model, random_data, clean_data, optimizer):
    with torch.no_grad():
        sse = F.mse_loss
        output = model(random_data)
        a_loss = sse(output, clean_data)
        test_loss = sse(output, clean_data, reduction='sum')
        # print("Test Loss :%.4f Standard Error:%.4f"%(a_loss, test_loss))
    return test_loss

def train_test(model, random_data, noisy_data, clean_data, epochs,image):
    logger_train = Logger(’../logs/train/{}/{}’.format(image, 1))
    logger_test = Logger(’../logs/test/{}/{}’.format(image, 1))
    eta = 1e-02
    optimizer = optim.Adam(model.parameters(), lr=eta)
    train_loss = np.zeros((epochs))
    test_loss = np.zeros((epochs))


for epoch in range(1, max_epoch):

    # output = model(noisy_img)
    # loss = criterion(output, clean_img)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # test_loss = test(model, noisy_img, clean_img)
    #
    # print('loss:{:.4f}'.format(loss))



    loss_train, loss_output = train(model, random_data, noisy_data, optimizer)
    loss_test = test(model, noisy_img, clean_img, optimizer)
    train_loss[epochs-1] = loss_train
    test_loss[epochs-1] = loss_test
    info_train = {’loss’: loss_train.item()}
    info_test = {’loss’: loss_test.item()}

    for tag, value in info_train.items():
        logger_train.scalar_summary(tag, value, epoch + 1)
    for tag, value in info_test.items():
        logger_test.scalar_summary(tag, value, epoch + 1)

# Shows final result
out = model(eta)
out_img = out[0, 0, :, :].transpose(0,1).detach().numpy()

error1 = ((im - noise1)**2).sum()
error2 = ((im - out_img)**2).sum()

plt.figure(3)
plt.axis('off')

plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.title('Input')

plt.subplot(132)
plt.imshow(noise1, cmap='gray')
plt.title('SE {:.2f}'.format(error1))

plt.subplot(133)
plt.imshow(out_img, cmap='gray')
plt.title('SE {:.2f}'.format(error2))

plt.show()
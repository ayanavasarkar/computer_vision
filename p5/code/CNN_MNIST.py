import numpy as np
import scipy.io as spio
from skimage.feature import plot_matches

import torch
import torchvision
import torch.utils.data as utils
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def todict(matobj):

    dict = {}
    for strg in matobj.fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(path):
    return todict(spio.loadmat(path,
                               struct_as_record=False,
                               squeeze_me=True)['data'])

normal = loadmat("digits-scaled.mat")

x_data = normal['x']
x_data = x_data.reshape((28*28, -1)).T
y_data = normal['y']
print(x_data.shape)

train_x = x_data[normal['set']==1]
train_y = y_data[normal['set']==1]
train_x = train_x.reshape((train_x.shape[0],1,28,28))

val_x = x_data[normal['set']==2]
val_x = val_x.reshape((val_x.shape[0],1,28,28))
val_y = y_data[normal['set']==2]

test_x = x_data[normal['set']==3]
test_x = test_x.reshape((test_x.shape[0],1,28,28))
test_y = y_data[normal['set']==3]

train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).long()

val_x = torch.from_numpy(val_x).float()
val_y = torch.from_numpy(val_y).long()

test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).long()

train_dataset = utils.TensorDataset(train_x, train_y)
test_dataset = utils.TensorDataset(test_x, test_y)
val_dataset = utils.TensorDataset(val_x, val_y)

trainloader = utils.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
testloader = utils.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)
valloader = utils.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=2)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def val(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    epochs = 150

    model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.005)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(1, epochs + 1):
        train(model, device, trainloader, optimizer, epoch)
        test(model, device, testloader)



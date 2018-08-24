# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

import utils
from rednet import RedNet
# from unet import Unet

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

train_data_dir = 'data/train/'
batch_size = 16
print("load training images")
data = utils.np_from_imgs(train_data_dir, batch_size)
batch_num = len(data)

device = torch.device('cuda')
dtype = torch.cuda.FloatTensor
print("build unet")
net = RedNet().to(device)
net.apply(utils.weights_init)
optimizer = torch.optim.Adam(net.parameters())
criterion = nn.MSELoss()

def train(epoch):
    epoch_loss = 0.
    for j in range(1, batch_num+1):
        clean_data = data[j-1]
        sigma = np.random.uniform(0, 51)
        input_data = np.clip(clean_data + np.random.normal(scale=sigma, size=clean_data.shape), 0, 255)
        input_tensor = torch.from_numpy(input_data).type(dtype).to(device)
        optimizer.zero_grad()
        output_tensor = net(input_tensor)
        sigma = np.random.uniform(0, 51)
        target_data = np.clip(clean_data + np.random.normal(scale=sigma, size=clean_data.shape), 0, 255)
        target_tensor = torch.from_numpy(target_data).type(dtype).to(device)
        loss = criterion(output_tensor, target_tensor)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        print("Epoch[{}]({}/{}): loss: {:.4f}".format(epoch, j, batch_num+1, loss.item()))
    print("Epoch {} Complete: Avg. loss: {:.4f}".format(epoch, epoch_loss / batch_num))

def checkpoint(epoch):
    path = "checkpoint/unet_epoch_{}.pth".format(epoch)
    torch.save(net, path)
    print("Checkpoint saved to {}".format(path))


epoch = 100
for i in range(1, epoch+1):
    train(i)
    checkpoint(i)

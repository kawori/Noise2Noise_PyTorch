# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class RedNet(nn.Module):
    def __init__(self, in_channels=3):
        super(RedNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 128, 3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, padding=1), nn.ReLU())
        self.deconv2 = nn.ConvTranspose2d(128, 3, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        convs = []
        convs.append(self.conv1(x))
        for i in range(1, 15):
            convs.append(self.conv2(convs[i-1]))
        y = self.deconv1(convs[14]) + convs[13]
        for i in range(1, 14):
            y = self.deconv1(y)
            if i % 2 == 0:
                y += convs[13-i]
                y = self.relu(y)
        y = self.deconv2(y)
        return y

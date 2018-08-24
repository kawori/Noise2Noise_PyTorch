# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ConvTranspose2d(48, 48, 2, 2, output_padding=1)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ConvTranspose2d(96, 96, 2, 2)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(99, 64, 3, padding=1),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # print("input: ", x.shape)
        pool1 = self.block1(x)
        # print("pool1: ", pool1.shape)
        pool2 = self.block2(pool1)
        # print("pool2: ", pool2.shape)
        pool3 = self.block2(pool2)
        # print("pool3: ", pool3.shape)
        pool4 = self.block2(pool3)
        # print("pool4: ", pool4.shape)
        pool5 = self.block2(pool4)
        # print("pool5: ", pool5.shape)
        upsample5 = self.block3(pool5)
        # print("upsample5: ", upsample5.shape)
        concat5 = torch.cat((upsample5, pool4), 1)
        # print("concat5: ", concat5.shape)
        upsample4 = self.block4(concat5)
        # print("upsample4: ", upsample4.shape)
        concat4 = torch.cat((upsample4, pool3), 1)
        # print("concat4: ", concat4.shape)
        upsample3 = self.block5(concat4)
        # print("upsample3: ", upsample3.shape)
        concat3 = torch.cat((upsample3, pool2), 1)
        # print("concat3: ", concat3.shape)
        upsample2 = self.block5(concat3)
        # print("upsample2: ", upsample2.shape)
        concat2 = torch.cat((upsample2, pool1), 1)
        # print("concat2: ", concat2.shape)
        upsample1 = self.block5(concat2)
        # print("upsample1: ", upsample1.shape)
        concat1 = torch.cat((upsample1, x), 1)
        # print("concat1: ", concat1.shape)
        output = self.block6(concat1)
        # print("output: ", output.shape)
        return output

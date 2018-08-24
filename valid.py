# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from unet import Unet


def np_to_pil(np_imgs):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    img_num = np_imgs.shape[0]
    channel_num = np_imgs.shape[1]
    ar = np.clip(np_imgs*255, 0, 255).astype(np.uint8)

    pil_imgs = []
    for i in range(img_num):
        if channel_num == 1:
            img = ar[i][0]
        else:
            img = ar[i].transpose(1, 2, 0)
        pil_imgs.append(Image.fromarray(img))

    return pil_imgs


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
valid_data_dir = 'data/valid'
batch_size = 64
# img_size = 112
dataset = datasets.ImageFolder(
    root=valid_data_dir, transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, num_workers=2)

cuda = torch.device('cuda')
dtype = torch.cuda.FloatTensor
net = Unet().type(dtype)
net.load_state_dict(torch.load("checkpoint/unet.pth"))
for j, data in enumerate(dataloader):
    clean_data, _ = data
    sigma = np.random.uniform(0, 51)
    input_data = torch.clamp(clean_data + torch.zeros(clean_data.shape).normal_(std=sigma), 0, 1).type(dtype)
    input_data.to(cuda)
    pil_imgs = np_to_pil(input_data.detach().cpu().numpy())
    for i in range(batch_size):
        pil_imgs[i].save("res/noise/"+str(j*batch_size+i)+".jpg")
    output_data = net(input_data)
    pil_imgs = np_to_pil(output_data.detach().cpu().numpy())
    for i in range(batch_size):
        pil_imgs[i].save("res/denoise/"+str(j*batch_size+i)+".jpg")

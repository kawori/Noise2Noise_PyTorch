# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def np_from_imgs(img_dir, batch_size):
    img_paths = os.listdir(img_dir)
    img_num = len(img_paths)
    assert img_num % batch_size == 0
    batch_num = img_num // batch_size
    ret = []
    for i in range(batch_num):
        patch = []
        for j in range(batch_size):
            idx = i * batch_size + j
            img = np.array(Image.open(img_dir + img_paths[idx]))
            img = img.transpose(2, 0, 1).astype(np.float32)
            patch.append(img)
        ret.append(np.array(patch))
    return ret


def np_to_pil(np_imgs):
    img_num = np_imgs.shape[0]
    ar = np.clip(np_imgs * 255, 0, 255).astype(np.uint8)

    pil_imgs = []
    for i in range(img_num):
        img = ar[i].transpose(1, 2, 0)
        pil_imgs.append(Image.fromarray(img))

    return pil_imgs


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

# img = np.array(Image.open("test.jpg")).astype(np.float32)
# sigma = np.random.uniform(0, 51)
# print("sigma1:", sigma)
# img1 = np.clip(img + np.random.normal(scale=sigma, size=img.shape), 0, 255).astype(np.uint8)
# plt.figure()
# plt.imshow(img1)
# sigma = np.random.uniform(0, 51)
# print("sigma2:", sigma)
# img2 = np.clip(img + np.random.normal(scale=sigma, size=img.shape), 0, 255).astype(np.uint8)
# plt.figure()
# plt.imshow(img2)
# plt.show()

# -*- coding: utf-8 -*-
# Time    : 19-3-13 下午6:07
# Author  : zlich
# Filename: determine The image type.py
'''
将自己的图片转化为model可载入的数据图片类型
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps

path = os.getcwd() + '/predict_image/'
files = os.listdir(path)

# print(files)
for file in files:
    if file.endswith('.jpg'):
        with Image.open(path + file) as img:
            img_L = img.convert('L')
            # plt.figure()
            # plt.imshow(img_L)
            img_M = PIL.ImageOps.invert(img_L)
            # plt.figure()
            # plt.imshow(img_M)
            img_M.save(file[0:-4]+'.jpg')

            # img1 = np.array(img)
            # print('origin:', img1.shape)
            # img_C = np.array(img.convert('L'))
            # print(img_C.shape)
            # plt.figure()
            # plt.imshow(img1)
            # plt.figure()
            # plt.imshow(img_C)

plt.show()

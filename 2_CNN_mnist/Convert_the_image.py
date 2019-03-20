# -*- coding: utf-8 -*-
# Time    : 19-3-13 下午6:07
# Author  : zlich
# Filename: determine The image type.py
'''
反转图像颜色，将自己的图片转化为model可载入的数据图片类型
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

            # 反转色彩
            img_M = PIL.ImageOps.invert(img_L)
            # plt.figure()
            # plt.imshow(img_M)
            # 保存经过转化后的图像
            img_M.save(file[0:-4] + '.jpg')

# plt.show()

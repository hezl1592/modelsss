# -*- coding: utf-8 -*-
# Time    : 19-3-14 上午10:00
# Author  : zlich
# Filename: getdata.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm


# 数据集照片的预览
def datasets_pic(path, categories=['Dog', 'Cat']):
    '''
    数据集照片的展示
    :param path: where the datasets are
    :return: Nothing
    '''

    for category in categories:
        path_image = os.path.join(path, category)
        # print(path_image)
        plt.figure('{} 数据集预览'.format(category), figsize=(8, 8))
        plt.axis('off')
        for i, file in enumerate(os.listdir(path_image)):
            # print(file)
            if i < 9:
                with Image.open(os.path.join(path_image, file)) as img:
                    plt.subplot(3, 3, i + 1)
                    plt.imshow(img)
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlabel('The picture is: {}'.format(file))
            else:
                break
    plt.show()


def pic_resize(path, resize_path, Image_size=100):
    categories = ['Dog', 'Cat']
    categories = ['Cat']
    for category in categories:
        path_image = os.path.join(path, category)
        # print(path_image)
        for file in tqdm(os.listdir(path_image)):
            # print(file)
            if file.endswith('.jpg'):
                with Image.open(os.path.join(path_image, file)) as img:
                    if img.mode != 'RGB':
                        print(file)
                    else:
                        out = img.resize((Image_size, Image_size))
                        out.save(os.path.join(resize_path, '{}_resize'.format(category)) + '/' + file)


# 用于展示data中的图片以及对应的标签, 验证shuffle的有效性
def figshow(array):
    n = np.random.randint(0, 400)
    array = array[n:n + 9, :]
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(array[i, 0])
        plt.xlabel('label: {}'.format(array[i, 1]))
        plt.xticks([])
        plt.yticks([])


def get_train_eval_data(path):
    categories = ['Cat_resize', 'Dog_resize']
    data = []
    x = []
    y = []

    # label: 0:Cat; 1:Dog
    for label, category in enumerate(categories):
        # print('---------the process progress----------')
        path_image = os.path.join(path, category)
        # print(path_image)

        for file in tqdm(os.listdir(path_image)):
            # print(file)
            if file.endswith('.jpg'):
                with Image.open(os.path.join(path_image, file)) as img:
                    img = img.convert('L')
                    img_np = np.array(img)
                    # img = img_np[]
                    # print(img_np.shape)
                    data.append([img_np, label])

    data = np.array(data)
    # print(data.shape)
    figshow(data)

    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    data = data[index, :]
    print(data.shape)
    figshow(data)

    for i in range(data.shape[0]):
        x.append(data[i, 0])
        y.append(data[i, 1])

    x = np.array(x)/255.0
    y = np.array(y)/255.0
    x = x.reshape(-1, 100, 100, 1)
    print(x.shape)
    print(y.shape)

    return x, y


if __name__ == "__main__":
    # 数据集路径
    path = os.path.join(os.getcwd(), 'data')

    # 预览数据集
    # datasets_pic(path=path)

    # resize 原始数据集图片
    # pic_resize(path, path)

    # resize数据集预览
    # datasets_pic(path=path, categories=['Cat_resize', 'Dog_resize'])

    # x, y = get_train_eval_data(path=path)
    # print(x.shape)
    # print(x[0].shape)




    # feature, label = get_train_eval_data(path=path)
    # print(feature[200].shape)
    # print()
    plt.show()

# -*- coding: utf-8 -*-
# Time    : 19-3-13 下午3:44
# Author  : zlich
# Filename: getdata.py

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image as Image
import os


# 下载mnist数据集，并且加载
# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()


def load_data(path='data/mnist'):
    '''
    加载本地路径中的mnist数据
    :param path: 路径位置
    :return: x,y的训练集与测试集
    '''
    x_train = np.load(path+'/'+'x_train.npy')
    y_train = np.load(path+'/'+'y_train.npy')
    x_test =np.load(path+'/'+'x_test.npy')
    y_test =np.load(path+'/'+'y_test.npy')

    return x_train, y_train, x_test, y_test


def load_predictdata(path=os.getcwd()+'/predict/'):
    '''
    load 自己的数据图片，将之处理为可读取的x_predict
    :param path:
    :return:
    '''
    files = os.listdir(path)
    # print(files)
    size = len(files)
    x_predict = np.zeros((size, 28, 28))
    # print(x_predict.shape)
    for i, file in enumerate(files):
        with Image.open(path+file) as img:
            img_N = np.array(img)
            x_predict[i,:,:] = img_N

    # print(x_predict)
    # print(x_predict.shape)

    return x_predict


# 主函数为校验数据的处理
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    load_predictdata()

    plt.figure('训练集数据展示', figsize=(8, 8))
    plt.axis('off')
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(x_train[i, :,:])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('The number is: {}'.format(y_train[i]))


    plt.show()

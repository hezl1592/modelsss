# -*- coding: utf-8 -*-
# Time    : 19-3-14 下午4:17
# Author  : zlich
# Filename: model_main.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from date_process import get_train_eval_data
import os


def model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(256, (3, 3), input_shape=(100, 100, 1)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(256, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64))

    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    return model


if __name__ == "__main__":
    # 数据集路径
    path = os.path.join(os.getcwd(), 'data')
    x, y = get_train_eval_data(path)
    model = model()
    model.fit(x, y, batch_size=32, epochs=4, validation_split=0.1)


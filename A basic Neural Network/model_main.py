# -*- coding: utf-8 -*-
# Time    : 19-3-13 下午4:35
# Author  : zlich
# Filename: model_main.py

import tensorflow as tf
from getdata import load_data
from getdata import load_predictdata
import numpy as np
import matplotlib.pyplot as plt


# 简单的全连接神经网络模型
def model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# 预测结果展示
def predict_result(x_predict, predictions):
    plt.figure('预测结果展示', figsize=(10, 10))
    plt.axis('off')
    for i, n in enumerate(predictions):
        plt.subplot(4, 4, i + 1)
        plt.imshow(x_predict[i, :, :])
        n = np.argmax(n)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('The number is: {}'.format(n))
    plt.savefig('测试集2.jpg')
    plt.show()


# 主函数
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    x_train_pro = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
    x_test_pro = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)
    # print(type(x_train))

    model = model()
    print('-----------training proess------------')
    model.fit(x_train_pro, y_train, epochs=3)

    print('-----------evaluation process----------')
    val_loss, val_acc = model.evaluate(x_test_pro, y_test)
    print(val_loss)
    print(val_acc)

    # # 采用自己写的图片
    # x_predict = load_predictdata()
    # x_predict_ev = tf.keras.utils.normalize(x_predict, axis=1).reshape(x_predict.shape[0], -1)
    # print(x_predict.shape, x_predict_ev.shape)

    # 采用测试集中的
    n = np.random.randint(0, 10000)
    x_predict = x_test[n:n + 16, :, :]
    print(x_predict.shape)
    x_predict_ev = x_test_pro[n:n + 16, :]

    predictions = model.predict(x_predict_ev)
    predict_result(x_predict, predictions)

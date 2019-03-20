# -*- coding: utf-8 -*-
# Time    : 19-3-17 下午2:50
# Author  : zlich
# Filename: eval.py
import tensorflow as tf
import new_model as model
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import os
from getdata import load_data

CHECK_POINT_DIR = os.path.join(os.getcwd(), 'checkpoint/')


def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 28, 28, 1])

        logit = model.inference(image, 1, 10)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[28, 28, 1])

        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('Reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split(
                    '/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            print(prediction)
            # if max_index == 1:
            #     result = ('this is cat rate: %.6f' % (prediction[:, 1]))
            # else:
            #     result = ('this is dog rate: %.6f' % (prediction[:, 0]))
            return max_index


if __name__ == '__main__':
    eval_pic_path = os.path.join(os.getcwd(), 'predict')


    # 利用测试集来检验预测结果
    x_train, y_train, x_test, y_test = load_data()
    print(x_test[1].shape)

    n = np.random.randint(0, int(x_test.shape[0]/2))
    plt.figure('预测结果展示: n={}'.format(n), figsize=(8, 8))
    # plt.axis('off')
    for i, img in enumerate(x_test[n:n+9]):
        plt.subplot(3, 3, i + 1)
        img = img.reshape((28,28))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        img = np.array(img).reshape((28, 28, 1))
        result = evaluate_one_image(img)
        plt.xlabel('real result:{}, predict: {}'.format(y_test[n+i], result))
    plt.show()

    # 利用自己的图片检验预测效果
    # plt.figure('预测结果展示', figsize=(8, 8))
    # # plt.axis('off')
    # for i, file in enumerate(os.listdir(eval_pic_path)):
    #     # print(file)
    #     if i < 9:
    #         with Image.open(os.path.join(eval_pic_path, file)) as img:
    #             plt.subplot(3, 3, i + 1)
    #             plt.imshow(img)
    #             plt.xticks([])
    #             plt.yticks([])
    #             # img = img.resize(1, 28, 28)
    #             print(np.array(img).shape)
    #             img = np.array(img).reshape((28, 28,1))
    #             result = evaluate_one_image(img)
    #             plt.xlabel('{}'.format(result))
    #     else:
    #         break
    # plt.show()

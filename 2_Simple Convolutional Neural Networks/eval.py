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

CHECK_POINT_DIR = os.path.join(os.getcwd(), 'checkpoint/')


def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 100, 100, 3])

        logit = model.inference(image, 1, 2)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[100, 100, 3])

        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('Reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            print(prediction)
            if max_index == 1:
                result = ('this is cat rate: %.6f' % (prediction[:, 1]))
            else:
                result = ('this is dog rate: %.6f' % (prediction[:, 0]))
            return result


if __name__ == '__main__':
    eval_pic_path = os.path.join(os.getcwd(), 'test_image')


    for file in os.listdir(eval_pic_path):
        with Image.open(os.path.join(eval_pic_path, file)) as image:
            plt.figure('{}'.format(file))
            plt.imshow(image)
            image = image.resize([100, 100])
            image = np.array(image)
            result = evaluate_one_image(image)
            plt.xlabel('{}'.format(result))
    plt.show()


    # plt.figure('预测结果展示', figsize=(8, 8))
    # plt.axis('off')
    # for i, file in enumerate(os.listdir(eval_pic_path)):
    #     # print(file)
    #     if i < 9:
    #         with Image.open(os.path.join(eval_pic_path, file)) as img:
    #             plt.subplot(3, 3, i + 1)
    #             plt.imshow(img)
    #             plt.xticks([])
    #             plt.yticks([])
    #             img = img.resize([100, 100])
    #             img = np.array(img)
    #             result = evaluate_one_image(img)
    #             plt.xlabel('{}'.format('{}'.format(result)))
    #     else:
    #         break
    # plt.show()

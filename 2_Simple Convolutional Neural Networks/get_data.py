# -*- coding: utf-8 -*-
# Time    : 19-3-17 下午2:30
# Author  : zlich
# Filename: get_data.py
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def get_files(file_path):
    class_train = []
    label_train = []
    for label, category in enumerate(os.listdir(file_path)):
        print(label, category)
        for pic_name in tqdm(os.listdir(file_path + category)):
            class_train.append(file_path + category + '/' + pic_name)
            label_train.append(label)
    temp = np.array([class_train, label_train])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    # class is 1 2 3 4 5
    label_list = [int(i) for i in label_list]
    return image_list, label_list


def get_batches(image, label, resize_w, resize_h, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)
    queue = tf.train.slice_input_producer([image, label])
    label = queue[1]
    image_temp = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_temp, channels=3)
    # resize image
    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)

    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)
    images_batch = tf.cast(image_batch, tf.float32)
    labels_batch = tf.reshape(label_batch, [batch_size])
    return images_batch, labels_batch


if __name__ == "__main__":
    img_path = os.path.join(os.getcwd(), 'data_new/')
    print(img_path)
    image_list, label_list = get_files(img_path)

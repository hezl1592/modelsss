# -*- coding: utf-8 -*-
# Time    : 19-3-17 下午2:40
# Author  : zlich
# Filename: train.py
import os
import tensorflow as tf
from get_data import get_batches, get_files
import new_model as model
import numpy as np

LOG_DIR = os.path.join(os.getcwd(), 'logs/')
CHECK_POINT_DIR = os.path.join(os.getcwd(), 'checkpoint/')
img_path = os.path.join(os.getcwd(), 'data_new/')

train, train_label = get_files(img_path)

train_batch, train_label_batch = get_batches(train, train_label, 100, 100, 10, 20)

train_logits = model.inference(train_batch, 10, 2)

train_loss = model.losses(train_logits, train_label_batch)

train_op = model.trainning(train_loss, 0.001)

train_acc = model.evaluation(train_logits, train_label_batch)

summary_op = tf.summary.merge_all()

sess = tf.Session()
train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()

threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    # 尝试加载存在的训练参数
    print('Reading checkpoints...')
    ckpt = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        global_step = 0
        print('No checkpoint file found')

    for step in np.arange(10000):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

        if step % 100 == 0:
            print('Step %d, train loss=%.6f, train accuracy = %.6f' % (int(global_step) + step, tra_loss, tra_acc))
            # print('Step %d, train loss=%.6f, train accuracy = %.6f' % (step, tra_loss, tra_acc))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
        if (step+1) % 1000 == 0:
            checkpoint_path = os.path.join(CHECK_POINT_DIR, 'model_ckpt')
            saver.save(sess, checkpoint_path, global_step=int(global_step)+step)
            # saver.save(sess, checkpoint_path, global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training')
finally:
    coord.request_stop()
coord.join(threads)

# -*- coding: utf-8 -*-
# Time    : 19-3-17 下午2:40
# Author  : zlich
# Filename: train.py
import os
import tensorflow as tf
from getdata import get_batches, load_data
import new_model as model
import numpy as np

LOG_DIR = os.path.join(os.getcwd(), 'logs/')
CHECK_POINT_DIR = os.path.join(os.getcwd(), 'checkpoint/')
# data_path = os.path.join(os.getcwd(), 'data_new/')

x_train, y_train, x_test, y_test = load_data()

train_batch, train_label_batch = get_batches(x_train, y_train, 20, 20)

train_logits = model.inference(train_batch, 20, 10)

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
    # 尝试加载存在的训练参数，继续训练
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
        if (step+1) % 100 == 0:
            checkpoint_path = os.path.join(CHECK_POINT_DIR, 'model_ckpt')
            saver.save(sess, checkpoint_path, global_step=int(global_step)+step)
            # saver.save(sess, checkpoint_path, global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training')
finally:
    coord.request_stop()
    coord.join(threads)

import tensorflow as tf
import os
from model.net3d_model import net_3d
import config as cfg
import time
import numpy as np
from data import DataReader

reader = DataReader('val')
pointcloud_train, true_box_train = reader.provide(cfg.test_batch_size)
training = tf.placeholder(dtype=tf.bool, shape=[], name='training')
pointcloud = tf.placeholder(dtype=tf.float32, shape=[None, cfg.pc_size, cfg.pc_channel], name='pointclouds')
model = net_3d()
pred = model.inference(pointcloud, training)
box, score, cls = model.predict(pred)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
saver = tf.train.Saver()
result = []
with tf.Session(config= config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint(cfg.model_dir))
    graph = tf.get_default_graph()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(cfg.test_num):
        start_time = time.time()
        pointcloud_value, true_box_value = sess.run([pointcloud_train, true_box_train])
        box_value, score_value, class_value = sess.run([box, score, cls], {pointcloud: pointcloud_value, training: False})
        labels_per_pc = 10
        pc = np.reshape(pointcloud_value, (250000, 3))
        true_box = np.reshape(true_box_value, (200, 14))
        pred_box_score = np.concatenate((box_value, score_value), axis=-1)
        pred_box_score = np.reshape(pred_box_score, (400, 10))
        result.append(np.array([pc, true_box, pred_box_score]))
        print(np.shape(pc))
        print(np.shape(true_box))
        print(np.shape(pred_box_score))
        duration = time.time() - start_time
        format_str = ('Tested pointcloud {} using {} seconds')
        print(format_str.format(i, duration))
    coord.request_stop()
    coord.join(threads)
print(np.shape(result))
result = np.array(result)
np.save('YoloResult.npy', result)












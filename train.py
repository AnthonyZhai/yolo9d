import tensorflow as tf
import os
from model.net3d_model import net_3d
import config as cfg
import time
from data import DataReader
#prepare data
reader = DataReader('train')
pointcloud_train, true_box_train = reader.provide(cfg.batch_size)
model = net_3d()
yolo_out = [int(cfg.pc_height/(8 * cfg.reso_height)), int(cfg.pc_width/(8 * cfg.reso_width)), cfg.num_anchors, 10 + cfg.num_classes]
training = tf.placeholder(dtype=tf.bool, shape=[], name='training')
pointcloud = tf.placeholder(dtype=tf.float32, shape=[None, cfg.pc_size, cfg.pc_channel], name='pointclouds')
true_box = tf.placeholder(dtype=tf.float32, shape=[None, yolo_out[0], yolo_out[1], yolo_out[2], yolo_out[3]], name='labels')
pred_box = model.inference(pointcloud, training)
loss, xyz_t, abg_p = model.loss(pred_box, true_box, cfg.anchors, training=True)
global_step = tf.Variable(0, trainable = False, name='global_step')
#decayed_learning_rate = learning_rate * 0.1 ^ (global_step / 3000)
learning_rate = tf.train.exponential_decay(cfg.learning_rate, global_step, 3000, 0.1)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss = loss, global_step = global_step)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
saver = tf.train.Saver()
with tf.Session(config = config) as sess:
    ckpt = tf.train.get_checkpoint_state(cfg.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('restore model', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        #sess.run(tf.local_variables_initializer())
    else:
        sess.run(init)
    summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    for epoch in range(cfg.epoch):
        steps = int(cfg.train_num / cfg.batch_size)
        for step in range(steps):
            start_time = time.time()
            pointcloud_value, true_box_value = sess.run([pointcloud_train, true_box_train])
            merged = tf.summary.merge_all()
            train_loss, true_xyz, pred_abg, merged_summary, _ = sess.run([loss, xyz_t, abg_p, merged, train_op], {pointcloud: pointcloud_value, true_box: true_box_value, training: True})
            duration = time.time() - start_time
            examples_per_sec = float(duration) / cfg.batch_size
            format_str = ('Epoch {} step {},  train loss = {} ( {} examples/sec; {} ''sec/batch)')
            print(format_str.format(epoch, step, train_loss, examples_per_sec, duration))
            # tf.Summary(value = [tf.Summary.Value(tag = "train loss", simple_value = train_loss)])
            summary_writer.add_summary(summary = merged_summary, global_step = epoch * steps + step)
            summary_writer.flush()
        if epoch % 2 == 0:
            checkpoint_path = os.path.join(cfg.model_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step = global_step)
    coord.request_stop()
    coord.join(threads)

# import tensorflow as tf
# a = tf.constant([[3,2],[7,6],[4,9]], dtype=tf.float32)
# reordered = tf.gather(a, tf.nn.top_k(a[:,0], k=3).indices)  # 按照输入的第一个维度排序，选取top3的值
# with tf.Session() as sess:
#     #print(sess.run(tf.argmax(a, axis=-1)))
#     print(sess.run(reordered))
#     #print(sess.run(tf.gather(a, tf.argmax(a, axis=-1))))
#     #print(sess.run(tf.a# rgmax(a, axis=-1)))
#     print(sess.run(tf.reduce_max(a, axis=-1)))
import numpy as np
import config as cfg
import tensorflow as tf
L = int(cfg.pc_depth / cfg.reso_depth)
H = int(cfg.pc_height / cfg.reso_height)
W = int(cfg.pc_width / cfg.reso_width)
a = [2,4,6,8,10,12,14,16,18,20,22,24]
a1 = tf.constant(a, dtype=tf.float32)
a1 = tf.reshape(a1, [2, 2, 3])
a2 = [2,2,2]
a3 = tf.constant([1,1,1,1,1,1], dtype=tf.float32)
a3 = tf.reshape(a3, [2, 1, 3])
b1 = tf.constant([0.17287,0.24858,0.91877,
-0.11899,0.04760,1.03221,
-0.17643,0.19726,0.89236,
0.35052,0.18755,0.89108,
0.14771,-0.06510,1.03229,])
b1 = tf.reshape(b1, [5,3])
b2 = cfg.offset
b3 = tf.constant([cfg.reso_width, cfg.reso_height])

with tf.Session() as sess:
    print(sess.run(a1/a2-a3))
    grid_y = tf.tile(tf.reshape(tf.range(5), [-1, 1, 1, 1]), [1, 5, 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(5), [1, -1, 1, 1]), [5, 1, 1, 1])

    grid = tf.concat([grid_x, grid_y], axis=-1)
    grid = tf.cast(grid, tf.float32)
    print('grid:' ,sess.run(tf.shape(grid)))
    print(sess.run(grid))
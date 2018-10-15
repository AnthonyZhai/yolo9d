import tensorflow as tf
import numpy as np
import config as cfg
from utils import *


class net_3d:

    def __init__(self):
        self.rL = cfg.reso_depth
        self.rH = cfg.reso_height
        self.rW = cfg.reso_width
        self.L = int(cfg.pc_depth / self.rL)
        self.H = int(cfg.pc_height / self.rH)
        self.W = int(cfg.pc_width / self.rW)
        self.num_anchors = cfg.num_anchors
        self.num_classes = cfg.num_classes
        self.anchors = cfg.anchors
        self.classes = cfg.classes
        self.batch_size = cfg.batch_size
        self.nms_threshold = cfg.nms_threshold
        self.obj_threshold = cfg.obj_threshold

    def conv2d_bn_relu_layer(self, inputs, filters, kernel_size, name, padding=1, strides=1, norm_decay=0.997,
                             norm_epsilon=1e-5, training=True):
        conv = self.conv2d_layer(inputs=inputs,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 name=name,
                                 padding=padding,
                                 strides=strides)
        bn = tf.layers.batch_normalization(inputs=conv,
                                           momentum=norm_decay,
                                           epsilon=norm_epsilon,
                                           center=True,
                                           scale=True,
                                           training=training,
                                           fused=False)
        return tf.nn.leaky_relu(bn, alpha=0.1)

    def conv2d_layer(self, inputs, filters, kernel_size, name, strides=1, padding=1):
        padded = tf.pad(tensor=inputs,
                        paddings=[[0, 0], [padding, padding], [padding, padding], [0, 0]],
                        mode='CONSTANT')
        return tf.layers.conv2d(
            inputs=padded,
            filters=filters,
            kernel_size=kernel_size,
            name=name,
            strides=(strides, strides),
            padding='VALID'
        )

    def YOLO(self, inputs, training):
        with tf.variable_scope('yolo_network') as scope:
            conv = self.conv2d_bn_relu_layer(inputs=inputs, filters=64, kernel_size=3, name="conv_1", training=training)

            conv = self.conv2d_bn_relu_layer(inputs=conv, filters=128, kernel_size=3, name="conv_2", training=training)

            maxpool1 = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2, name="maxpooling_1")

            conv = self.conv2d_bn_relu_layer(inputs=maxpool1, filters=128, kernel_size=3, name="conv_3",
                                             training=training)

            conv = self.conv2d_bn_relu_layer(inputs=conv, filters=64, kernel_size=1, padding=0, name="conv_4",
                                             training=training)

            conv = self.conv2d_bn_relu_layer(inputs=conv, filters=128, kernel_size=3, name="conv_5", training=training)

            maxpool2 = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2, name="maxpooling_2")

            conv = self.conv2d_bn_relu_layer(inputs=maxpool2, filters=256, kernel_size=3, name="conv_6",
                                             training=training)

            conv = self.conv2d_bn_relu_layer(inputs=conv, filters=128, kernel_size=1, padding=0, name="conv_7",
                                             training=training)

            conv = self.conv2d_bn_relu_layer(inputs=conv, filters=256, kernel_size=3, name="conv_8", training=training)

            maxpool3 = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2, name="maxpooling_3")

            conv = self.conv2d_bn_relu_layer(inputs=maxpool3, filters=512, kernel_size=3, name="conv_9",
                                             training=training)

            conv = self.conv2d_bn_relu_layer(inputs=conv, filters=256, kernel_size=1, padding=0, name="conv_10",
                                             training=training)

            conv = self.conv2d_bn_relu_layer(inputs=conv, filters=512, kernel_size=3, name="conv_11", training=training)

            conv = self.conv2d_bn_relu_layer(inputs=conv, filters=256, kernel_size=1, padding=0, name="conv_12",
                                             training=training)

            conv = self.conv2d_bn_relu_layer(inputs=conv, filters=512, kernel_size=3, name="conv_13", training=training)

            output = self.conv2d_layer(inputs=conv, filters=cfg.filters, kernel_size=3, padding=1, name="conv_14")

        return output

    def fc_layer(self, inputs, out_dem, name):
        dense = tf.layers.dense(inputs=inputs,
                                units=out_dem,
                                activation=tf.nn.relu,
                                name=name + '_dense')
        return tf.layers.batch_normalization(inputs=dense,
                                             fused=True,
                                             name=name + '_bn')

    def vfe_layer2(self, inputs, in_dim, out_dim, name):
        # inputs [K,T,6]
        pointwised = self.fc_layer(inputs, out_dim / 2, name)

        aggregated = tf.reduce_max(input_tensor=pointwised,
                                   axis=1,
                                   keepdims=True,
                                   name=name + '_maxpl')
        repeated = tf.tile(input=aggregated,
                           multiples=[1, cfg.maxium_points_per_cell, 1])

        convatenaed = tf.concat([pointwised, repeated],
                                axis=2,
                                name=name + 'conc')
        return convatenaed

    def vfe_layer(self, inputs, in_dim, out_dim, name):
        # inputs [B,K,T,6]
        pointwised = self.fc_layer(inputs, out_dim / 2, name)

        aggregated = tf.reduce_max(input_tensor=pointwised,
                                   axis=2,
                                   keepdims=True,
                                   name=name + '_maxpl')
        repeated = tf.tile(input=aggregated,
                           multiples=[1, 1, cfg.maxium_points_per_cell, 1])

        convatenaed = tf.concat([pointwised, repeated],
                                axis=3,
                                name=name + 'conc')
        return convatenaed

    def ext(self, point_cloud):
        batch_size = np.shape(point_cloud)[0]
        dict_list = []
        for b in range(batch_size):
            dict_list.append(self.extension_v2(point_cloud[b]))
        feature_list = []
        coordinate_list = []
        for i, voxel_dict in zip(range(batch_size), dict_list):
            feature_list.append(voxel_dict['feature_buffer'])
            coordinate = voxel_dict['coordinate_buffer']
            coordinate_list.append(
                np.pad(coordinate, ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))

        feature = np.concatenate(feature_list)
        coordinate = np.concatenate(coordinate_list)
        return feature, coordinate

    def extension_v2(self, point_cloud):
        # point_cloud [N, 3] & N = 250000 now
        voxel_size = np.array([self.rL, self.rH, self.rW], dtype=np.float32)
        # grid_size = np.array([self.L, self.H, self.W], dtype=np.int64)

        np.random.shuffle(point_cloud)
        shifted_coord = point_cloud[..., :3] + cfg.offset

        voxel_index = np.floor(shifted_coord[..., ::-1] / voxel_size).astype(np.int32)

        # bound_x = np.logical_and(
        #     voxel_index[:, 2] >= 0, voxel_index[:, 2] < self.W)
        # bound_y = np.logical_and(
        #     voxel_index[:, 1] >= 0, voxel_index[:, 1] < self.H)
        # bound_z = np.logical_and(
        #     voxel_index[:, 0] >= 0, voxel_index[:, 0] < self.L)
        #
        # bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
        #
        # voxel_index = voxel_index[bound_box]
        # point_cloud = point_cloud[bound_box]

        coordinate_buffer = np.unique(voxel_index, axis=0)

        K = coordinate_buffer.shape[0]
        T = cfg.maxium_points_per_cell
        # [K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=(K), dtype=np.int64)

        # [K, T, 6] feature buffer as described in the paper
        feature_buffer = np.zeros(shape=(K, T, 6), dtype=np.float32)

        # build a reverse index for coordinate buffer
        index_buffer = {}
        for i in range(K):
            index_buffer[tuple(coordinate_buffer[i])] = i

        for voxel, point in zip(voxel_index, point_cloud):
            index = index_buffer[tuple(voxel)]
            number = number_buffer[index]
            if number < T:
                feature_buffer[index, number, :3] = point
                number_buffer[index] += 1

        feature_buffer[..., -3:] = feature_buffer[..., :3] - \
                                   feature_buffer[..., :3].sum(axis=1, keepdims=True) / number_buffer.reshape(K, 1, 1)
        voxel_dict = {'feature_buffer': feature_buffer,
                      'coordinate_buffer': coordinate_buffer}
        return voxel_dict

    def extension(self, point_cloud):
        # point_cloud tensor [B,250000,3]
        voxel_size = np.array([self.rL, self.rH, self.rW], dtype=np.float32)
        # grid_size = np.array([self.L, self.H, self.W], dtype=np.int64)

        np.random.shuffle(point_cloud)

        # reverse the point cloud coordinate (X, Y, Z) -> (Z, Y, X)
        voxel_index = np.floor(point_cloud[..., ::-1] / voxel_size).astype(np.int32)

        # [B, K, 3] coordinate buffer as described in the paper
        coordinate_buffer = np.unique(voxel_index, axis=1)

        B = point_cloud.shape[0]
        K = coordinate_buffer.shape[1]
        T = cfg.maxium_points_per_cell
        # [B, K, 1] store number of points in each voxel grid
        number_buffer = np.zeros(shape=(B, K), dtype=np.int64)

        # [B, K, T, 6] feature buffer as described in the paper
        feature_buffer = np.zeros(shape=(B, K, T, 6), dtype=np.float32)

        # bound_x = np.logical_and(
        #     voxel_index[..., 2] >= 0, voxel_index[..., 2] < grid_size[2])
        # bound_y = np.logical_and(
        #     voxel_index[..., 1] >= 0, voxel_index[..., 1] < grid_size[1])
        # bound_z = np.logical_and(
        #     voxel_index[..., 0] >= 0, voxel_index[..., 0] < grid_size[0])
        #
        # bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
        #
        # point_cloud = point_cloud[bound_box]
        # voxel_index = voxel_index[bound_box]

        for batch in range(B):
            # build a reverse index for coordinate buffer
            index_buffer = {}
            for i in range(K):
                index_buffer[tuple(coordinate_buffer[batch][i])] = i

            for voxel, point in zip(voxel_index[batch], point_cloud[batch]):
                index = index_buffer[tuple(voxel)]
                number = number_buffer[batch, index]
                if number < T:
                    feature_buffer[batch, index, number, :3] = point
                    number_buffer[batch, index] += 1
            coordinate_buffer = np.pad(coordinate_buffer[batch], ((0, 0), (1, 0)), mode='constant',
                                       constant_values=batch)

        feature_buffer[..., -3:] = feature_buffer[..., :3] - \
                                   feature_buffer[..., :3].sum(axis=2, keepdims=True) / number_buffer.reshape(B, K, 1,
                                                                                                              1)

        # feature_buffer = tf.convert_to_tensor(feature_buffer)
        # coordinate_buffer = tf.convert_to_tensor(coordinate_buffer)
        # for batch in range(B):
        #     coordinate_buffer = np.pad(coordinate_buffer[batch], ((0, 0), (1, 0)), mode='constant', constant_values=batch)
        return feature_buffer, coordinate_buffer

    def FLN(self, vfe, coordinate):
        # vfe [K,T,6]  coordinate [K,3]
        # vfe, coordinate = tf.py_func(self.extension_v2, [inputs[batch]], [tf.float32, tf.int32])
        # vfe.set_shape([None, cfg.maxium_points_per_cell, 6])
        # coordinate = tf.reshape(coordinate, [-1, 3])

        for index in range(cfg.num_vfes):
            vfe = self.vfe_layer2(vfe, cfg.vfe[2 * index], cfg.vfe[2 * index + 1], name='vfe' + str(index + 1))
        """
        vfe1 = self.vfe_layer(ext, 16, name='vfe1')

        vfe2 = self.vfe_layer(vfe1, 16, name='vfe2')

        vfe3 = self.vfe_layer(vfe2, 16, name='vfe3')
        """
        fc = self.fc_layer(vfe, out_dem=cfg.vfe[-1], name='fln')

        voxelwised = tf.reduce_max(input_tensor=fc,
                                   axis=1,
                                   name='fln_maxpl')
        return tf.scatter_nd(coordinate, voxelwised, [cfg.batch_size, self.L, self.H, self.W, cfg.vfe[-1]])

        # vfe, coordinate = tf.py_func(self.extension, [inputs], [tf.float32, tf.int32])
        #
        # vfe.set_shape([self.batch_size, None, cfg.maxium_points_per_cell, 6])
        # coordinate = tf.reshape(coordinate, [self.batch_size, -1, 4])
        # #coordinate.set_shape([self.batch_size, None, 4])
        # for index in range(cfg.num_vfes):
        #     vfe = self.vfe_layer(vfe, cfg.vfe[2*index], cfg.vfe[2*index+1], name='vfe'+str(index+1))
        # """
        # vfe1 = self.vfe_layer(ext, 16, name='vfe1')
        #
        # vfe2 = self.vfe_layer(vfe1, 16, name='vfe2')
        #
        # vfe3 = self.vfe_layer(vfe2, 16, name='vfe3')
        # """
        # fc = self.fc_layer(vfe, out_dem=cfg.vfe[-1], name='fln')
        #
        # voxelwised = tf.reduce_max(input_tensor=fc,
        #                        axis=2,
        #                        name='fln_maxpl')
        # return tf.scatter_nd(coordinate, voxelwised, [self.batch_size, self.L, self.H, self.W, cfg.vfe[-1]])

    def inference(self, inputs, training=True):
        # inputs [B,N,3]

        feature, coordinate = tf.py_func(self.ext, [inputs], [tf.float32, tf.int32])
        feature = tf.reshape(feature, [-1, cfg.maxium_points_per_cell, 6])
        coordinate = tf.reshape(coordinate, [-1, 4])
        voxelwised = self.FLN(feature, coordinate)

        # voxelwised [B,D,H,W,C]
        transposed = tf.transpose(voxelwised, [0, 2, 3, 1, 4])
        transhaped = tf.reshape(tensor=transposed,
                                shape=[-1,
                                       self.H,
                                       self.W,
                                       self.L * cfg.vfe[-1]])

        out = self.YOLO(transhaped, training)

        return out


    def yolo_head(self, features, anchors, input_shape, training):
        """
        Introduction
        ------------
            根据不同大小的feature map做多尺度的检测，三种feature map大小分别为13x13x1024, 26x26x512, 52x52x256
        Parameters
        ----------
            feats: 输入的特征feature map
            anchors: 针对不同大小的feature map的anchor
            num_classes: 类别的数量
            input_shape: 图像的输入大小，一般为416
            trainging: 是否训练，用来控制返回不同的值
        Returns
        -------
        """
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, self.num_anchors, 6])
        grid_size = tf.shape(features)[1:3]
        predictions = tf.reshape(features, [-1, grid_size[0], grid_size[1], self.num_anchors, self.num_classes + 10])
        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])

        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)
        # 将x,y坐标归一化为占416的比例
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        box_z = tf.sigmoid(predictions[..., 2:3])
        box_xyz = tf.concat([box_xy, box_z], axis=-1)
        # 将w,h也归一化为占416的比例
        box_whl = tf.exp(predictions[..., 3:6]) * anchors_tensor[..., 0:3] / input_shape[::-1]
        box_abg = tf.exp(predictions[..., 6:9]) * anchors_tensor[..., 3:6] / (2 * np.pi)
        # box_abg = predictions[..., 6:9] + anchors_tensor[..., 3:6]
        #可能以后需要改成box_abg = predictions[..., 6:9] + anchors_tensor[..., 3:6] / (2 * ni.pi)  取决于label该项是否为相对值
        box_confidence = tf.sigmoid(predictions[..., 9:10])
        box_class_probs = tf.sigmoid(predictions[..., 10:])
        if training == True:
            return grid, predictions, box_xyz, box_whl, box_abg
        return box_xyz, box_whl, box_abg, box_confidence, box_class_probs

    def loss(self, yolo_output, y_true, anchors, ignore_thresh=.5, training=True):
        # y_true [B,H,W,#B,10+K]
        """
        Introduction
        ------------
            yolo模型的损失函数
        Parameters
        ----------
            yolo_output: yolo模型的输出
            y_true: 经过预处理的真实标签，shape为[batch, grid_size, grid_size, 5 + num_classes]
            anchors: yolo模型对应的anchors
            num_classes: 类别数量
            ignore_thresh: 小于该阈值的box我们认为没有物体
        Returns
        -------
            loss: 每个batch的平均损失值
            accuracy
        """
        loss = 0
        input_shape = [cfg.pc_depth, cfg.pc_height, cfg.pc_width]
        # input_shape = [self.L, self.H, self.W]
        grid_shapes = [tf.cast(tf.shape(yolo_output)[1:3], tf.float32)]
        xyz_loss_per_batch = 0
        whl_loss_per_batch = 0
        abg_loss_per_batch = 0
        confidence_loss_per_batch = 0
        class_loss_per_batch = 0
        for index in range(self.num_anchors):
            # 只有负责预测ground truth box的grid对应的为1, 才计算相对应的loss
            # object_mask的shape为[batch_size, grid_size, grid_size, #anchor, 1]
            # confidence
            object_mask = y_true[..., 9:10]
            # 类别概率 onehotvector
            class_probs = y_true[..., 10:]
            # 将yolo输出预处理为有意义
            grid, predictions, pred_xyz, pred_whl, pred_abg = self.yolo_head(yolo_output, anchors, input_shape, training)

            pred_box = tf.concat([pred_xyz, pred_whl, pred_abg], axis=-1)
            # Bx13x13x1x3

            #raw_true_xy = y_true[..., :2] * grid_shapes[::-1] - grid
            # 10.10 修改
            raw_true_xy = y_true[..., :2] * grid_shapes[::-1] - grid
            raw_true_z = y_true[..., 2:3]
            raw_true_xyz = tf.concat([raw_true_xy, raw_true_z], axis=-1)
            object_mask_bool = tf.cast(object_mask, dtype=tf.bool)
            #10.11 删除* input_shape[::-1]从y_true[..., 3:6] / anchors[index * 6 + 0: index * 6 + 3]后面
            raw_true_whl = tf.log(
                tf.where(tf.equal(y_true[..., 3:6] / anchors[index * 6 + 0: index * 6 + 3] * input_shape[::-1], 0),
                         tf.ones_like(y_true[..., 3:6]),
                         y_true[..., 3:6] / anchors[index * 6 + 0: index * 6 + 3] * input_shape[::-1]))
            ##########  modified. ##########
            raw_true_abg = tf.log(
                tf.where(tf.equal(y_true[..., 6:9] / anchors[index * 6 + 3: index * 6 + 6] * (2 * np.pi), 0),
                         tf.ones_like(y_true[..., 6:9]),
                         y_true[..., 6:9] / anchors[index * 6 + 3: index * 6 + 6] * (2 * np.pi)))
            #raw_true_abg = y_true[..., 6:9] - anchors[index * 6 + 3: index * 6 + 6]


            # 该系数是用来调整box坐标loss的系数
            # box_loss_scale = 2 - y_true[..., 3:4] * y_true[..., 4:5] * y_true[..., 5:6]
            ignore_mask = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

            def loop_body(internal_index, ignore_mask):
                # true_box的shape为[box_num, 9]
                y = y_true[internal_index]
                mask = object_mask_bool[internal_index, ..., 0]
                # mask = tf.expand_dims(mask, axis=-2)
                true_box = tf.boolean_mask(y, mask)
                # true_box = tf.reshape(true_box, [tf.shape(y)[0], tf.shape(y)[1], -1, tf.shape(y)[3]])

                true_boxes = true_box[..., 0:9] * [cfg.pc_width, cfg.pc_height, cfg.pc_depth,
                                                   cfg.pc_width, cfg.pc_height, cfg.pc_depth,
                                                   2*np.pi, 2*np.pi, 2*np.pi]

                pred_boxes = pred_box[internal_index, ..., 0:9] * [cfg.pc_width, cfg.pc_height, cfg.pc_depth,
                                                                   cfg.pc_width, cfg.pc_height, cfg.pc_depth,
                                                                   2*np.pi, 2*np.pi, 2*np.pi]
                # input type [H,W,#B,9] [N,9]
                # output [H,W,#B,N]
                iou = tf.py_func(cell_iou_batch, [pred_boxes, true_boxes], tf.float32)
                # 计算每个true_box对应的预测的iou最大的box
                best_iou = tf.reduce_max(iou, axis=-1)
                # HxWxBx1
                ignore_mask = ignore_mask.write(internal_index, tf.cast(best_iou < ignore_thresh, tf.float32))
                return internal_index + 1, ignore_mask

            # BXHxWx#B
            _, ignore_mask = tf.while_loop(
                lambda internal_index, ignore_mask: internal_index < tf.shape(yolo_output)[0], loop_body,
                [0, ignore_mask])

            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, axis=-1)

            # 计算四个部分的loss
            # xyz_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels = raw_true_xyz, logits = predictions[..., 0:3])
            obj_scale = 1
            noobj_scale = 0.5
            box_loss_scale = 2 - y_true[..., 3:4] * y_true[..., 4:5] * y_true[..., 5:6]
            xyz_loss = object_mask * box_loss_scale * tf.square(raw_true_xyz - tf.sigmoid(predictions[..., 0:3]))
            whl_loss = object_mask * box_loss_scale * tf.square(raw_true_whl - predictions[..., 3:6])

            abg_loss = object_mask * box_loss_scale * tf.square(raw_true_abg - predictions[..., 6:9])
            # print(abg_loss.eval())
            confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,logits=predictions[...,9:10]) \
                              + (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=predictions[..., 9:10]) * ignore_mask

            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 10:],
                                                                                           logits=predictions[..., 10:])

            xyz_loss_per_batch += tf.reduce_sum(xyz_loss) / tf.cast(tf.shape(yolo_output)[0], tf.float32)
            whl_loss_per_batch += tf.reduce_sum(whl_loss) / tf.cast(tf.shape(yolo_output)[0], tf.float32)
            abg_loss_per_batch += tf.reduce_sum(abg_loss) / tf.cast(tf.shape(yolo_output)[0], tf.float32)
            confidence_loss_per_batch += tf.reduce_sum(confidence_loss) / tf.cast(tf.shape(yolo_output)[0], tf.float32)
            class_loss_per_batch += tf.reduce_sum(class_loss) / tf.cast(tf.shape(yolo_output)[0], tf.float32)
        loss = xyz_loss_per_batch + whl_loss_per_batch + abg_loss_per_batch + 0 * class_loss_per_batch + confidence_loss_per_batch
        tf.summary.scalar('xyz_loss', xyz_loss_per_batch)
        tf.summary.scalar('whl_loss', whl_loss_per_batch)
        tf.summary.scalar('abg_loss', abg_loss_per_batch)
        tf.summary.scalar('confidence_loss', confidence_loss_per_batch)
        tf.summary.scalar('class_loss', class_loss_per_batch)
        tf.summary.scalar('batch_loss', loss)
        return loss, raw_true_xyz, predictions[..., 6:9]

    def boxes_and_scores(self, feats, anchors, input_shape):
        """
        Introduction
        ------------
            将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
        Parameters
        ----------
            feats: yolo输出的feature map
            anchors: anchor的位置
            class_num: 类别数目
            input_shape: 输入大小
            image_shape: 图片大小
        Returns
        -------
            boxes: 物体框的位置
            boxes_scores: 物体框的分数，为置信度和类别概率的乘积
        """
        box_xyz, box_whl, box_abg, box_confidence, box_class_probs = self.yolo_head(feats, anchors, input_shape, False)
        box_loc = box_xyz * [cfg.pc_width, cfg.pc_height, cfg.pc_depth] - cfg.offset
        box_dem = box_whl * [cfg.pc_width, cfg.pc_height, cfg.pc_depth]
        box_ang = box_abg * [2*np.pi,2*np.pi,2*np.pi]-[np.pi,np.pi,np.pi]

        boxes = tf.concat([box_loc, box_dem, box_ang], axis=-1)
        boxes = tf.reshape(boxes, [-1, 9])

        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, self.num_classes])
        box_confidence = tf.reshape(box_confidence, [-1, 1])

        return boxes, box_confidence, box_scores

    def eval(self, yolo_outputs, max_boxes=20):
        """
        Introduction
        ------------
            根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
        Parameters
        ----------
            yolo_outputs: yolo模型输出
            image_shape: 图片的大小
            max_boxes:  最大box数量
        Returns
        -------
            boxes_: 物体框的位置
            scores_: 物体类别的概率
            classes_: 物体类别
        """

        input_shape = [self.W, self.H, self.L]
        # 获取每个预测box坐标和box的分数，score计算为置信度x类别概率
        boxes, confidence, box_scores = self.boxes_and_scores(yolo_outputs, self.anchors, input_shape)
        return boxes, confidence, box_scores
        # # 判断属于哪个类别
        # box_classes = tf.argmax(box_scores, axis=-1)
        # # 从K个类得分（置信度）中找到最大类得分作为该预测的得分
        # box_class_scores = tf.reduce_max(box_scores, axis=-1)
        # # 找出置信度大于阈值的index mask
        # prediction_mask = box_class_scores >= self.obj_threshold
        # o_boxes = tf.boolean_mask(boxes, prediction_mask)
        # o_scores = tf.boolean_mask(box_class_scores, prediction_mask)
        # o_classes = tf.boolean_mask(box_classes, prediction_mask)
        #
        # return o_boxes, o_scores, o_classes

    def predict(self, inputs):
        """
        Introduction
        ------------
            构建预测模型
        Parameters
        ----------
            inputs: 处理之后的输入图片
            image_shape: 图像原始大小
        Returns
        -------
            boxes: 物体框坐标
            scores: 物体概率值
            classes: 物体类别
        """
        boxes, scores, classes = self.eval(inputs)
        return boxes, scores, classes

import config as cfg
import numpy as np
import os
import tensorflow as tf
from utils import *
import config as cfg

class DataReader:
    def __init__(self, mode):
        self.data_dir = cfg.data_dir
        self.label_dir = cfg.label_dir
        self.mode = mode
        self.record = os.path.join(self.data_dir, self.mode + '.tfrecords')
        self.train_val_list = {'train': cfg.train_list, 'val': cfg.val_list}
        self.L = int(cfg.pc_depth / cfg.reso_depth)
        self.H = int(cfg.pc_height / cfg.reso_height)
        self.W = int(cfg.pc_width / cfg.reso_width)
        self.num_anchors = cfg.num_anchors
        self.num_classes = cfg.num_classes
        self.anchors = cfg.anchors
        self.classes = cfg.classes
        self.tfrecord = self.record
        if not os.path.exists(self.tfrecord):
            self.convert_to_tfrecord(self.data_dir)
    """
    def pc_preprocess(self, pc_data):
        # pc_data [N, 900000]
        pc_data = tf.reshape(pc_data, [-1, 300000, 3])
        pc_grid = np.zeros([self.L, self.H, self.W, self.max_points, 6], dtype=np.float32)
        point_counter = np.zeros([self.L, self.H, self.W, 1])
        for num in range(tf.shape(pc_data)[0]):
            for point in pc_data[num]:
                #point = tf.reshape(point)
                D = point[2]//cfg.reso_depth
                H = point[1]//cfg.reso_height
                W = point[0]//cfg.reso_width
                cD = D + cfg.reso_depth//2
                cH = H + cfg.reso_height//2
                cW = W + cfg.reso_width//2
                num = point_counter[D][H][W]
                if num < self.max_points:
                    pc_grid[D][H][W][num][0:3] = point
                    pc_grid[D][H][W][num][3:6] = point - [cW, cH, cD]
                    point_counter[D][H][W] += 1

        return pc_grid
    """

    def label_preprocess(self, true_boxes):
        # label [T,10] anchor [N,6]
        y_true = np.zeros([self.W // 8, self.H // 8, self.num_anchors, 10 + self.num_classes], dtype=np.float32)
        anchors = np.reshape(self.anchors, [self.num_anchors, 6])
        labels = true_boxes[..., 3:9] * [cfg.pc_width, cfg.pc_height, cfg.pc_depth, 2*np.pi, 2*np.pi, 2*np.pi]
        iou = np.zeros([np.shape(labels)[0], np.shape(anchors)[0]], dtype=np.float32)
        for gt in range(np.shape(iou)[0]):
            label = np.concatenate((np.array([0, 0, 0]), labels[gt]))
            for an in range(np.shape(iou)[1]):
                anchor = np.concatenate((np.array([0, 0, 0]), anchors[an]))
                iou[gt][an] = cell_iou(anchor, label)

        best_anchor = np.argmax(iou, axis=-1)

        # 计算最优IOU [B,N,6]和[B,T,6] best

        for t, n in enumerate(best_anchor):
            i = np.floor(true_boxes[t, 0] * self.W // 8 ).astype('int32')
            j = np.floor(true_boxes[t, 1] * self.H // 8).astype('int32')
            c = true_boxes[t, 9].astype('int32')
            y_true[j, i, n, 0:9] = true_boxes[t, 0:9]
            y_true[j, i, n, 9] = 1
            y_true[j, i, n, 10 + c] = 1

        return y_true

    def read_data(self):
        """
        Introduction
        ------------
            读取数据集图片路径和对应的标注
        Parameters
        ----------
            data_file: 文件路径
        """
        pc_data = []
        label_data = []
        #准备pc路径
        with open(self.train_val_list[self.mode], encoding='utf-8') as file:
            for line in file.readlines():
                pc_data.append(line.strip('\n'))
        #准备label数据
        for pc in pc_data:
            name = os.path.splitext(os.path.basename(pc))[0]
            with open(os.path.join(self.data_dir, 'label' +name.split('data')[1] + '.txt')) as label:
                label_lines = []
                for line in label.readlines():
                    line = line.strip('\n').split(',')
                    # line[:3] = (line[:3] + cfg.offset) / [cfg.pc_width, cfg.pc_height, cfg.pc_depth]
                    # line[3:6] = line[3:6] / [cfg.pc_width, cfg.pc_height, cfg.pc_depth]
                    label_lines.append(line)
            label_data.append(label_lines)
        return pc_data, label_data

    def convert_to_tfrecord(self, tfrecord_path):
        """
        Introduction
        ------------
            将图片和boxes数据存储为tfRecord
        Parameters
        ----------
            tfrecord_path: tfrecord文件存储路径
        """
        output_file = os.path.join(tfrecord_path, self.mode + '.tfrecords')
        pc_data, label_data = self.read_data()
        record_writer = tf.python_io.TFRecordWriter(output_file)
        for index, pc_name in enumerate(pc_data):
            file = open(os.path.join(tfrecord_path, pc_name), 'r')
            pc = []
            for line in file.readlines():
                pc.append(line.strip('\n').split(','))
            pc = np.array(pc, dtype=np.float32)
            pc = np.reshape(pc, [-1]).tolist()

            boxes = []
            for box in label_data[index]:
                boxes.append(box)
            boxes = np.array(boxes, dtype=np.float32)
            boxes = np.reshape(boxes, [-1, 10])
            boxes[:, :3] = (boxes[:, :3] + cfg.offset) / [cfg.pc_width, cfg.pc_height, cfg.pc_depth]
            boxes[:, 3:6] = boxes[:, 3:6] / [cfg.pc_width, cfg.pc_height, cfg.pc_depth]
            boxes[:, 6:9] = (boxes[:, 6:9] + np.pi) / (2 * np.pi)
            boxes = self.label_preprocess(boxes)
            boxes = np.reshape(boxes, [-1]).tolist()

            if np.isnan(boxes).sum() != 0:
                print(pc_name)
                print(np.isnan(boxes).sum())
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'pc': tf.train.Feature(float_list=tf.train.FloatList(value=pc)),
                    'label': tf.train.Feature(float_list=tf.train.FloatList(value=boxes))
                }
            ))
            record_writer.write(example.SerializeToString())
            print('Processed {} of {} point cloud'.format(index + 1, len(pc_data)))
            file.close()
        record_writer.close()

    def parser(self, serialized_example):
        """
        Introduction
        ------------
            解析tfRecord数据
        Parameters
        ----------
            serialized_example: 序列化的每条数据
        """
        features = tf.parse_single_example(
            serialized_example,
            features = {
                'pc' : tf.FixedLenFeature([cfg.pc_size, cfg.pc_channel], dtype = tf.float32),
                'label' : tf.FixedLenFeature([self.H // 8, self.W // 8, cfg.num_anchors, 10 + cfg.num_classes], dtype = tf.float32)
            }
        )
        pc = features['pc']
        #pc = tf.reshape(pc, [300000, 3])
        label = features['label']
        #label = tf.reshape(label, [self.H / 8, self.W / 8, cfg.num_anchors, 10 + cfg.num_classes])
        #label = tf.py_func(self.label_preprocess, [label], tf.float32)
        #true_label.set_shape([None, 10])
        return pc, label

    def provide(self, batch_size):
        filename_queue = tf.train.string_input_producer([self.tfrecord])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        true_pc, true_label = self.parser(serialized_example)
        true_pc.set_shape([cfg.pc_size, cfg.pc_channel])
        true_label.set_shape([self.H // 8, self.W // 8, cfg.num_anchors, 10 + cfg.num_classes])
        images, boxes_true = tf.train.shuffle_batch([true_pc, true_label], batch_size = batch_size, capacity = 10 * batch_size, num_threads = 4, min_after_dequeue = 2 * batch_size)
        return images, boxes_true

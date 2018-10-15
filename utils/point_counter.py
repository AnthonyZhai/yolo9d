import config as cfg
import numpy as np
import os
import matplotlib.pyplot as plt
import random
reso_width=0.01
reso_height=0.01
reso_depth=0.01

pc_width=3.04
pc_height=2.56
pc_depth=2.5

grid_h = int(pc_height/reso_height)
grid_w = int(pc_width/reso_width)
grid_d = int(pc_depth/reso_depth)

data_dir = "./data"


def read_data():
    pc_files = []
    # 准备pc路径
    with open('train.txt', encoding='utf-8') as file:
        for line in file.readlines():
            pc_files.append(line.strip('\n'))
    # 准备label数据
    pc_counter = np.zeros([20, grid_d, grid_h, grid_w], dtype=np.int32)

    for i, file_name in enumerate(pc_files):
        pc_data = []
        with open(os.path.join(data_dir, file_name)) as file:
            for line in file.readlines():
                pc_data.append(line.strip('\n').split(','))
        pc = np.array(pc_data, dtype=np.float32)
        np.random.shuffle(pc)
        pc = pc + cfg.offset

        voxel_size = np.array([reso_depth, reso_height, reso_width], dtype=np.float32)

        voxel_index = np.floor(pc[..., ::-1] / voxel_size).astype(np.int32)

        for index in voxel_index:
            pc_counter[i][index[0]][index[1]][index[2]] += 1
    counter = pc_counter.reshape([-1, 1])
    mean = counter[counter > 0]
    mean = np.sort(mean)
    print(np.mean(mean))
    print(np.median(mean))
    print('-----------------\n')

read_data()


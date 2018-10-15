import numpy as np
import os
import config as cfg
data_dir = './data'
data = []
for name in os.listdir(data_dir):
    if 'label' in name:
        with open(os.path.join(data_dir, name)) as f:
            for line in f.readlines():
                split = line.strip('\n').split(',')
                nums = []
                for c in split:
                    nums.append(float(c))
                data.append(np.array(nums) + np.array([0,0,0,0,0,0,np.pi,np.pi,np.pi,0]))
data = np.array(data, dtype=np.float32)
print(np.shape(data))
mean_column = np.mean(data, axis=0)
print(mean_column * [cfg.pc_width, cfg.pc_height, cfg.pc_depth,
                     cfg.pc_width, cfg.pc_height, cfg.pc_depth,
                     1           , 1            , 1           , 1 ])

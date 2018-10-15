#3D point cloud size
pc_width=3.04   #304
pc_height=2.56  #256
pc_depth=2.5   #250
offset=[pc_width/2.0, pc_height/2.0, 0]
#3D voxel grid cell size
reso_width=0.076
reso_height=0.064
reso_depth=0.01

obj_threshold = 0.4
nms_threshold = 0.4

#maximun points_per_grid cell
maxium_points_per_cell = 90

#num of vfelayers
num_vfes = 3

#input&output of each vfelayers
vfe = [7, 16, 16, 16, 16, 16]
pc_size = 250000
pc_channel = 3

#num of classes
num_classes = 4
#classes
classes=['box', 'fileholder']
#num of anchors per grid cell
num_anchors = 4
#anchors 0.28028892  0.11940923  0.2421066 0.02678523 -0.03438516 -0.01164442
# anchors=[0.2688, 0.183, 0.1254, 0.7854, 1.5708, 1.5708,
#          0.2688, 0.183, 0.1254, 2.3562, 1.5708, 1.5708,
#          0.3052, 0.2129, 0.020, 0.7854, 1.5708, 1.5708,
#          0.3052, 0.2129, 0.020, 2.3562, 1.5708, 1.5708,
#          0.2978, 0.2563, 0.073, 0.7854, 1.5708, 1.5708,
#          0.2978, 0.2563, 0.073, 2.3562, 1.5708, 1.5708,
#          0.4719, 0.3198, 0.096, 0.7854, 1.5708, 1.5708,
#          0.4719, 0.3198, 0.096, 2.3562, 1.5708, 1.5708]
anchors=[0.7115, 0.4307, 0.5517, 3.5174, 2.7961, 3.4073,
         1.0178, 0.0655, 0.6185, 4.3953, 3.2480, 3.3545,
         0.7642, 0.3002, 0.5979, 3.2874, 3.0935, 3.2703,
         1.2430, 0.2714, 0.6615, 3.5213, 3.3997, 2.9740]
#last conv layer's filter of YOLO
#equals to B*(10+K)
filters= num_anchors * (10 + num_classes)
#train number
train_num = 20
test_num = 10
#batch size
batch_size = 4
test_batch_size = 1
#epoch
epoch = 50
#learning rate
learning_rate = 1e-3
#model directory
model_dir = './backup'
data_dir = './data'
label_dir = './label'
train_list = './train.txt'
val_list = './val.txt'

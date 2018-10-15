'''
Documentation, License etc.
@package FatFrustum2PC
'''

from viz_util import draw_rgbd_pointcloud
import pose_util
import numpy as np
import config as cfg
D=cfg.pc_depth
H=cfg.pc_height
W=cfg.pc_width

c = np.load("YoloResult.npy")

pc1=c[0,0]
print(pc1.shape)
gt1=c[0,1]
print(gt1.shape)
pred1=c[0,2]
print(pred1.shape)

pc=pc1
gt=gt1
pred=pred1

box3d_gt_list=[]
box3d_pred_list=[]


viz_all_pointCloud=True

for i in range(gt.shape[0]):
    if gt[i,9]==1:

        loc_gt=gt[i,:3]*[W,H,D]-cfg.offset
        box3d_dimensions_gt=gt[i,3:6]*[W,H,D]
        angles_gt=gt[i,6:9]*[2*np.pi,2*np.pi,2*np.pi]-[np.pi,np.pi,np.pi]
        box3d_gt = pose_util.compute_box_3d_euler_angel(loc_gt, box3d_dimensions_gt,angles_gt)

        box3d_gt_list.append(box3d_gt)
    if pred[i, 9] >=0.7:
        loc_pred = pred[i, :3]
        box3d_dimensions_pred = pred[i, 3:6]
        angles_pred = pred[i, 6:9]

        print('pred score: ',pred[i, 9])

        box3d_pred = pose_util.compute_box_3d_euler_angel(loc_pred, box3d_dimensions_pred,angles_pred)

        box3d_pred_list.append(box3d_pred)

print(len(box3d_pred_list))

if viz_all_pointCloud:
    seg=np.ones(pc.shape[0])
    print(seg.shape)
    draw_rgbd_pointcloud(pc, seg=seg,box3d_gt=box3d_gt_list, box3d_pred=box3d_pred_list)













import numpy as np
from mathutils import Euler

def compute_box_3d_euler_angel(inputs):
    location = inputs[:3]
    box3d_dimensions = inputs[3:6]
    angles = inputs[6:9]

    eul = Euler(angles)

    # 3d bounding box dimensions
    w, h, l = box3d_dimensions
    # 3d bounding box corners
    x_corners = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2]

    R = eul.to_matrix()

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + location[0]
    corners_3d[1, :] = corners_3d[1, :] + location[1]
    corners_3d[2, :] = corners_3d[2, :] + location[2]

    return np.transpose(corners_3d)


def compute_corner_diag(label):
    '''

    :param label: (24,)
    :return:
    '''
    corner_label = label.reshape(8, 3)

    corner_label_x = corner_label[:, 0]
    corner_label_y = corner_label[:, 1]
    corner_label_z = corner_label[:, 2]

    x_mean = np.mean(corner_label_x, axis=0)
    y_mean = np.mean(corner_label_y, axis=0)
    z_mean = np.mean(corner_label_z, axis=0)

    return 2 * np.sqrt(
        np.square(corner_label[0, 0] - x_mean) + np.square(corner_label[0, 1] - y_mean) + np.square(
            corner_label[0, 2] - z_mean))


def cell_iou(pred, label):
    '''

    :param pred:
    :param label:
    :return:
    '''
    pred = compute_box_3d_euler_angel(pred)
    label = compute_box_3d_euler_angel(label)

    diagonal = compute_corner_diag(label)
    corner_dist = pred - label
    corner_dist = np.linalg.norm(corner_dist, axis=1)
    corner_dist = np.sum(corner_dist, axis=0) / 8.0
    corner_dist_diag_ratio = corner_dist / diagonal
    confidece = get_gt_confidence(corner_dist_diag_ratio, 0.5, 1)
    return np.array(confidece, dtype=np.float32)

def cell_iou_batch(pred, label):
    '''

    :param pred: HXWXBX9
    :param label: Nx9 N #GTBOX
    :return:
    '''
    #input [H,W,#B,9]
    pred_shape = np.shape(pred)
    gt_shape=np.shape(label)
    result = np.zeros(([pred_shape[0],pred_shape[1],pred_shape[2],gt_shape[0]]), dtype=np.float32)
    for h in range(pred_shape[0]):
        for w in range(pred_shape[1]):
            for a in range(pred_shape[2]):
                #x y z w h l alpha beta gamma 9
                pre = pred[h][w][a]
                for gt in range(gt_shape[0]):
                    #9
                    labe = label[gt]
                    pred_3dbox = compute_box_3d_euler_angel_batch(pre)
                    label_3dbox = compute_box_3d_euler_angel_batch(labe)

                    diagonal = compute_corner_diag_batch(label_3dbox)
                    #8x3
                    corner_dist = pred_3dbox - label_3dbox
                    #8
                    corner_dist = np.linalg.norm(corner_dist, axis=-1)
                    #1
                    corner_dist = np.sum(corner_dist, axis=0) / 8.0
                    corner_dist_diag_ratio = corner_dist / diagonal

                    confidece = get_gt_confidence(corner_dist_diag_ratio, 0.5, 1)
                    result[h][w][a][gt] = confidece
                    #result[h][w][a][gt] = corner_dist_diag_ratio
    return result

def get_gt_confidence(offset_diagonal_ratio,alpha,dThreshold):
    '''
    Input:offset_diagonal_ratio BxMx8

    outPut:gt_confidence BxMx8
    '''

    dThreshold_tf = dThreshold*np.ones_like(offset_diagonal_ratio)

    offset_diagonal_ratio=np.where(np.greater_equal(offset_diagonal_ratio,dThreshold),dThreshold_tf,offset_diagonal_ratio)

    gt_confidence=(np.exp(alpha*(1-offset_diagonal_ratio/dThreshold)) -1)/(np.exp(alpha)-1)

    return gt_confidence

def compute_box_3d_euler_angel_batch(inputs):
    location = inputs[:3]
    box3d_dimensions = inputs[3:6]
    angles = inputs[6:9]
    eul = Euler(angles)

    # 3d bounding box dimensions
    w, h, l = box3d_dimensions
    # 3d bounding box corners
    x_corners = [-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]

    R = eul.to_matrix()

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + location[0]
    corners_3d[1, :] = corners_3d[1, :] + location[1]
    corners_3d[2, :] = corners_3d[2, :] + location[2]

    return np.transpose(corners_3d)


def compute_corner_diag_batch(label):
    corner_label = label.reshape(8, 3)
    corner_label_x = corner_label[:, 0]
    corner_label_y = corner_label[:, 1]
    corner_label_z = corner_label[:, 2]

    x_mean = np.mean(corner_label_x, axis=0)
    y_mean = np.mean(corner_label_y, axis=0)
    z_mean = np.mean(corner_label_z, axis=0)

    return 2 * np.sqrt(
        np.square(corner_label[0, 0] - x_mean) + np.square(corner_label[0, 1] - y_mean) + np.square(
            corner_label[0, 2] - z_mean))

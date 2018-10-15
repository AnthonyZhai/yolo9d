from mathutils import Vector, Matrix, Quaternion,Euler

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement


def draw_line(x1, y1, x2, y2, draw):
    xa = float(x1)
    ya = float(y1)
    xb = float(x2)
    yb = float(y2)
    draw.line([(xa, ya), (xb, yb)], "red")

def draw_3d_box(data, draw):
    draw_line(data[0][0], data[0][1], data[1][0], data[1][1], draw)
    draw_line(data[1][0], data[1][1], data[2][0], data[2][1], draw)
    draw_line(data[2][0], data[2][1], data[3][0], data[3][1], draw)
    draw_line(data[3][0], data[3][1], data[0][0], data[0][1], draw)

    draw_line(data[4][0], data[4][1], data[5][0], data[5][1], draw)
    draw_line(data[5][0], data[5][1], data[6][0], data[6][1], draw)
    draw_line(data[6][0], data[6][1], data[7][0], data[7][1], draw)
    draw_line(data[7][0], data[7][1], data[4][0], data[4][1], draw)

    draw_line(data[0][0], data[0][1], data[4][0], data[4][1], draw)
    draw_line(data[3][0], data[3][1], data[7][0], data[7][1], draw)
    draw_line(data[1][0], data[1][1], data[5][0], data[5][1], draw)
    draw_line(data[2][0], data[2][1], data[6][0], data[6][1], draw)

def draw_3d_bboxes(img_filepath, box3d_pixel):

    im_3d = Image.open(img_filepath)
    draw_3d = ImageDraw.Draw(im_3d)
    draw_3d_box(box3d_pixel, draw_3d)

    # save debug image showing bounding box correction
    del draw_3d
    new_img_3d_filepath = "/home/huichang/0000000002_color_labels.png"
    im_3d.save(new_img_3d_filepath)
    del im_3d

def draw_2d_bboxes(img_filepath, box2d_pixel):

    im_2d = Image.open(img_filepath)
    draw_2d = ImageDraw.Draw(im_2d)
    draw_line(box2d_pixel[0][0], box2d_pixel[0][1], box2d_pixel[1][0], box2d_pixel[1][1], draw_2d)
    draw_line(box2d_pixel[1][0], box2d_pixel[1][1], box2d_pixel[2][0], box2d_pixel[2][1], draw_2d)
    draw_line(box2d_pixel[2][0], box2d_pixel[2][1], box2d_pixel[3][0], box2d_pixel[3][1], draw_2d)
    draw_line(box2d_pixel[3][0], box2d_pixel[3][1], box2d_pixel[0][0], box2d_pixel[0][1], draw_2d)

    # save debug image showing bounding box correction
    del draw_2d
    new_img_2d_filepath = "/home/huichang/0000000003_color_labels.png"
    im_2d.save(new_img_2d_filepath)
    del im_2d

def get_3x4_RT_matrix(location,rotation):
    '''

    :param location:
    :param rotation:
    :return: RT_matrix
    '''
    rot = Quaternion(rotation)
    T = Vector(location)

    R = rot.to_matrix()

    # put into 3x4 matrix
    RT = Matrix((
        R[0][:] + (T[0],),
        R[1][:] + (T[1],),
        R[2][:] + (T[2],)
    ))
    return RT

# Return K*RT, K, and RT from the blender camera
def get_3x4_P_matrix(camIntrinsic,location,rotation):
    '''

    :param camIntrinsic:
    :param location:
    :param rotation:
    :return:proj, K, RT
    '''



    RT = get_3x4_RT_matrix(location,rotation)
    return Matrix(camIntrinsic) * RT, Matrix(camIntrinsic), RT


def get_box_from_pose(box3dMatrix,obj_loc,obj_rot,camIntrinsic):

    proj, K, RT = get_3x4_P_matrix(camIntrinsic,obj_loc, obj_rot)

    #print("box3dMatrix", box3dMatrix)

    coworld = np.array(box3dMatrix).T
    box3d_camera = np.dot(RT, coworld).T
    box3d_pixel = np.dot(proj, coworld)
    box3d_pixel = (box3d_pixel / box3d_pixel[2, :].T).T

    max_pixel = np.amax(box3d_pixel, axis=0)
    min_pixel = np.amin(box3d_pixel, axis=0)
    if min_pixel[0]<0:
        min_pixel[0]=0

    if min_pixel[1] < 0:
        min_pixel[1] = 0

    if max_pixel[0]>2*camIntrinsic[0,2]:
        max_pixel[0]=2*camIntrinsic[0,2]

    if max_pixel[1] > 2*camIntrinsic[1,2]:
        max_pixel[1] = 2*camIntrinsic[1,2]


    box2d_pixel = [
        [min_pixel[0], min_pixel[1]],
        [min_pixel[0], max_pixel[1]],
        [max_pixel[0], max_pixel[1]],
        [max_pixel[0], min_pixel[1]]
    ]

    # print("me_ob.matrix_world",me_ob.matrix_world)
   # print("box3d_pixel", box3d_pixel)
    #print("box3d_camera", box3d_camera)

    return box3d_camera,box3d_pixel,box2d_pixel



def getboxfrompose(object_meshes,obj_loc,obj_rot,camIntrinsic):
    objmodel = PlyData.read(object_meshes)

    min_x = np.min(objmodel.elements[0].data['x'])
    max_x = np.max(objmodel.elements[0].data['x'])

    min_y = np.min(objmodel.elements[0].data['y'])
    max_y = np.max(objmodel.elements[0].data['y'])

    min_z = np.min(objmodel.elements[0].data['z'])
    max_z = np.max(objmodel.elements[0].data['z'])

    proj, K, RT = get_3x4_P_matrix(camIntrinsic,obj_loc, obj_rot)

    box3dMatrix = [
        [min_x, min_y, min_z, 1],
        [min_x, max_y, min_z, 1],
        [max_x, max_y, min_z, 1],
        [max_x, min_y, min_z, 1],
        [min_x, min_y, max_z, 1],
        [min_x, max_y, max_z, 1],
        [max_x, max_y, max_z, 1],
        [max_x, min_y, max_z, 1]
    ]
    print("box3dMatrix", box3dMatrix)

    coworld = np.array(box3dMatrix).T
    box3d_camera = np.dot(RT, coworld).T
    box3d_pixel = np.dot(proj, coworld)
    box3d_pixel = (box3d_pixel / box3d_pixel[2, :].T).T

    max_pixel = np.amax(box3d_pixel, axis=0)
    min_pixel = np.amin(box3d_pixel, axis=0)

    if min_pixel[0]<0:
        min_pixel[0]=0

    if min_pixel[1] < 0:
        min_pixel[1] = 0

    if max_pixel[0]>2*camIntrinsic[0,2]:
        max_pixel[0]=2*camIntrinsic[0,2]

    if max_pixel[1] > 2*camIntrinsic[1,2]:
        max_pixel[1] = 2*camIntrinsic[1,2]

    box2d_pixel = [
        [min_pixel[0], min_pixel[1]],
        [min_pixel[0], max_pixel[1]],
        [max_pixel[0], max_pixel[1]],
        [max_pixel[0], min_pixel[1]]
    ]

    # print("me_ob.matrix_world",me_ob.matrix_world)
    print("box3d_pixel", box3d_pixel)
    print("box3d_camera", box3d_camera)

    return box3d_camera,box3d_pixel,box2d_pixel


def compute_box_3d(location, box3d_dimensions,quaternion_wxyz):
    ''' compute_box_3d based on location, box3d_dimensions, quaternion_wxyz
     input:
        box3d_dimensions:3    dimensions   3D object dimensions: height, width, length (in meters)
        location: 3    location     3D object location x,y,z in camera coordinates (in meters)
        quaternion_wxyz:4

         qs: (8,3) array of vertices for the 3d box in following order:
            7 -------- 6
           /|         /|
          4 -------- 5 .
          | |        | |
          . 3 -------- 2
          |/         |/
          0 -------- 1

     Returns:
         corners_3d: (8,3) array in rect camera coord.
    '''
    # 3d bounding box dimensions
    w, h,l= box3d_dimensions
    # 3d bounding box corners
    x_corners = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2]

    rot = Quaternion(quaternion_wxyz)#order wxyz
    R = rot.to_matrix()

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print(corners_3d.shape)
    corners_3d[0, :] = corners_3d[0, :] + location[0]
    corners_3d[1, :] = corners_3d[1, :] + location[1]
    corners_3d[2, :] = corners_3d[2, :] + location[2]
    # print('cornsers_3d: ', corners_3d)
    # only draw 3d bounding box for objs in front of the camera

    return  np.transpose(corners_3d)

def compute_box_2d_from_box3d_camera(box3d_camera,camIntrinsic):

    K = Matrix(camIntrinsic)

    box3d_pixel = np.dot(K, box3d_camera.T)
    box3d_pixel = (box3d_pixel / box3d_pixel[2, :].T).T

    max_pixel = np.amax(box3d_pixel, axis=0)
    min_pixel = np.amin(box3d_pixel, axis=0)

    if min_pixel[0]<0:
        min_pixel[0]=0

    if min_pixel[1] < 0:
        min_pixel[1] = 0

    if max_pixel[0]>2*camIntrinsic[0,2]:
        max_pixel[0]=2*camIntrinsic[0,2]

    if max_pixel[1] > 2*camIntrinsic[1,2]:
        max_pixel[1] = 2*camIntrinsic[1,2]

    box2d_pixel = [
        [min_pixel[0], min_pixel[1]],
        [min_pixel[0], max_pixel[1]],
        [max_pixel[0], max_pixel[1]],
        [max_pixel[0], min_pixel[1]]
    ]

    return box2d_pixel


def compute_box_3d_euler_angel(location, box3d_dimensions,angles):
    ''' compute_box_3d based on location, box3d_dimensions, Euler angles
    https://docs.blender.org/api/blender_python_api_current/mathutils.html#mathutils.Matrix
     input:
        box3d_dimensions:3    dimensions   3D object dimensions: height, width, length (in meters)
        location: 3    location     3D object location x,y,z in camera coordinates (in meters)
        angles:3    euler angles in order='XYZ' (3d vector) – Three angles, in radians.

         qs: (8,3) array of vertices for the 3d box in following order:
            7 -------- 6
           /|         /|
          4 -------- 5 .
          | |        | |
          . 3 -------- 2
          |/         |/
          0 -------- 1

     Returns:
         corners_3d: (8,3) array in rect camera coord.
    '''

    eul=Euler(angles)

    # 3d bounding box dimensions
    w, h,l= box3d_dimensions
    # 3d bounding box corners
    x_corners = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2]

    R = eul.to_matrix()

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print(corners_3d.shape)
    corners_3d[0, :] = corners_3d[0, :] + location[0]
    corners_3d[1, :] = corners_3d[1, :] + location[1]
    corners_3d[2, :] = corners_3d[2, :] + location[2]
    # print('cornsers_3d: ', corners_3d)
    # only draw 3d bounding box for objs in front of the camera

    return np.transpose(corners_3d)


def compute_corner_diag(corner_label):
    ''' Compute distance from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        corner_label: (B,24)
    Output:
        compute_distance_diagonal: (B,)
     '''
    batch_size = corner_label.shape[0]
    corner_label = corner_label.reshape(batch_size, 8, 3)
    corner_label_x = corner_label[:, :, 0]
    corner_label_y = corner_label[:, :, 1]
    corner_label_z = corner_label[:, :, 2]

    # B
    x_mean = np.mean(corner_label_x, axis=1)
    y_mean = np.mean(corner_label_y, axis=1)
    z_mean = np.mean(corner_label_z, axis=1)

    # np.linalg.norm(corner_label[:,:3]-corner_label[:,18:21],axis=1)

    return 2 * np.sqrt(
        np.square(corner_label[:, 0, 0] - x_mean) + np.square(corner_label[:, 0, 1] - y_mean) + np.square(
            corner_label[:, 0, 2] - z_mean))


def compute_corner_dist_diag_ratio(corner_pred, corner_label):
    ''' Compute distance from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        corner_pred: (B,24)
        corner_label: (B,24)
    Output:
        compute_distance_diagonal_ratio: (B,) compute_distance_diagonal_ratio
    '''

    batch_size = corner_pred.shape[0]
    # B
    # diagonal1=np.linalg.norm(corner_label[:,:3]-corner_label[:,18:21],axis=1)
    diagonal = compute_corner_diag(corner_label)

    # print('diagonal:',np.mean(diagonal-diagonal1))

    corner_dist = corner_pred - corner_label

    corner_dist = corner_dist.reshape(batch_size, 8, 3)
    # print('corner_dist:',corner_dist.shape)

    # Bx8
    corner_dist = np.linalg.norm(corner_dist, axis=2)
    # print('corner_dist:',corner_dist.shape)
    # B
    corner_dist = np.sum(corner_dist, axis=1) / 8.0

    corner_dist_diag_ratio = corner_dist / diagonal
    # print('corner_dist_diag_ratio:',corner_dist_diag_ratio.shape)

    return np.array(corner_dist_diag_ratio, dtype=np.float32)

if __name__=='__main__':
    camIntrinsic=[[528, 0.0, 320],[0.0, 528, 240],[0.0, 0.0, 1.0]]
    #camIntrinsic=[[619.444214, 0.0, 320],[0.0, 619.444336, 240],[0.0, 0.0, 1.0]]
    # #camExtrinsic=[-0.64111, 1.57144, 0.91089, 0.304, 0.255, -0.581, -0.711]
    #
    objpose_loc=[0.057806834094189324, -0.07366779032379833, 0.9684937784445177]
    objpose_rot=[0.9351808038693202, -0.3529547984653495, 0.023851878207472785, 0.017054683245150783]


    #objpose_loc=[-0.08029206825970772, 0.0710684671513715, 0.850568308652238]
    #objpose_rot=[-0.15380871374369712, 0.4082138599630501, -0.36110613697385374, 0.8242006320977904]

    #objpose_loc=[0.25918472883098165, -0.0830076128114241, 0.8943312131291252]
    #objpose_rot=[0.26840275193825286, 0.8362473210529089, 0.25839097675506334, 0.40234870934970135]

    object_meshes='/home/huichang/Desktop/LabelFusion/data/object-meshes/box_001.ply'
    #objmodel=PlyData.read('/home/huichang/Desktop/LabelFusion/data/object-meshes/box_001.ply')
    # #me_ob = bpy.data.objects['box_003']
    # #print("location", me_ob.location)

    cuboid_camera,box3d_pixel,box2d_pixel=getboxfrompose(object_meshes,objpose_loc,objpose_rot,camIntrinsic)
    draw_3d_bboxes("/home/huichang/0000000001_color_labels.png", box3d_pixel)
    draw_2d_bboxes("/home/huichang/0000000001_color_labels.png", box2d_pixel)

''' Visualization code for point clouds and 3D bounding boxes with mayavi.
Modified by Charles R. Qi
Date: September 2017
Ref: https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py
'''

import numpy as np
import mayavi.mlab as mlab
import mayavi
from tvtk.api import tvtk

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    #draw points
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=None, mode='point', colormap = 'gnuplot', scale_factor=1, figure=fig)
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    #draw axis
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

def draw_lidar(pc, color=None, fig=None, bgcolor=(0,0,0), pts_scale=1, pts_mode='point', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=pts_color, mode=pts_mode, colormap = 'gnuplot', scale_factor=pts_scale, figure=fig)
    
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    
    #draw axis
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)

    # draw fov (todo: update to real sensor spec.)
    fov=np.array([  # 45 degree
        [20., 20., 0.,0.],
        [20.,-20., 0.,0.],
    ],dtype=np.float64)
    
    mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
   
    # draw square region
    TOP_Y_MIN=-20
    TOP_Y_MAX=20
    TOP_X_MIN=0
    TOP_X_MAX=40
    TOP_Z_MIN=-2.0
    TOP_Z_MAX=0.4
    
    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    
    #mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=2, draw_text=False, text_scale=(1,1,1), color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    ''' 
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n] 
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def draw_rgbd_pointcloud(points3d,seg=None,seg_color=None,box3d_gt=None,box3d_pred=None,gt_color=(1,0,0),pred_color=(0,0,1),\
                         draw_text=False, text_scale=(1,1,1),color_list=None,camera_axis=True,save_name=None):

    if save_name is not None:
        mlab.options.offscreen = True

    figall = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(640, 480))
    if seg is not None:
        if seg_color is not None:
            nodes = mlab.points3d(points3d[:, 0], points3d[:, 1], points3d[:, 2], scale_factor=0.004,
                                  scale_mode='none', figure=figall)

            sc = tvtk.UnsignedCharArray()
            sc.from_array(seg_color)
            nodes.mlab_source.dataset.point_data.scalars = sc
            nodes.mlab_source.dataset.modified()
        else:
            nodes = mlab.points3d(points3d[:, 0], points3d[:, 1], points3d[:, 2], seg, scale_factor=0.004,
                                  scale_mode='none', colormap='gnuplot', figure=figall)



    else:
        nodes = mlab.points3d(points3d[:, 0], points3d[:, 1], points3d[:, 2],  mode='point', figure=figall)
        # nodes.glyph.scale_mode = 'scale_by_vector'
        sc = tvtk.UnsignedCharArray()
        sc.from_array(points3d[:, 3:6])
        nodes.mlab_source.dataset.point_data.scalars = sc
        nodes.mlab_source.dataset.modified()
        # camre_location = mlab.points3d(0, 0, 0,mode='point', figure=figall)

    #figall.scene.camera.position = [0, 0, 0]

    # camera_axis
    if camera_axis:
        mlab.quiver3d(0, 0, 0, 0, 0, 1, line_width=1, scale_factor=1, figure=figall)
        mlab.quiver3d(0, 0, 0, 0, 1, 0, line_width=1, scale_factor=1, figure=figall)
        mlab.quiver3d(0, 0, 0, 1, 0, 0, line_width=1, scale_factor=1, figure=figall)

    if box3d_gt is not None:
        draw_gt_boxes3d(box3d_gt, fig=figall,color=gt_color,draw_text=draw_text, text_scale=text_scale,color_list=color_list)

    if box3d_pred is not None:
        draw_gt_boxes3d(box3d_pred, fig=figall,color=pred_color,draw_text=draw_text, text_scale=text_scale,color_list=color_list)
    mlab.orientation_axes()

    #View 3dBox
    mlab.view(azimuth=180, elevation=190, focalpoint=[0.2,0, 0],distance=1.2, figure=figall)

    #View seg
    # mlab.view(azimuth=70, elevation=230, focalpoint=[0,0,0],distance=1, figure=figall)
    #
    # figall.scene.camera.position = [0.5094065699609567, -0.37414143582103904, -0.16310331266395872]
    # figall.scene.camera.focal_point = [-0.00448428033927202, -0.019859567181363474, 0.06892466310025992]
    # figall.scene.camera.view_angle = 30.0
    # figall.scene.camera.view_up = [-0.1517561654542513, 0.37800573712006946, -0.9132807503451144]
    # figall.scene.camera.clipping_range = [0.0017838578690604107, 1.7838578690604106]
    # figall.scene.camera.compute_view_plane_normal()
    # figall.scene.render()

    if save_name is not None:
        mlab.savefig(filename=save_name,figure=figall)

    else:
        mlab.show()

    input

if __name__=='__main__':
    pc = np.loadtxt('kitti_sample_scan.txt')
    fig = draw_lidar(pc)
    mlab.show()
    mlab.savefig('pc_view.jpg', figure=fig)
    raw_input()
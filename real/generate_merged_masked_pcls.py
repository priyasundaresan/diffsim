import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import cv2
import argparse
import pcl

def plot_pointclouds(pcls, title=""):
    fig = plt.figure(figsize=(10, 5))
    titles=['curr', 'ref']
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for i,points in enumerate(pcls):
        #ax = fig.add_subplot(1, 2, i+1, projection='3d')
        x, y, z = points.T
        ax.scatter3D(x, y, z, s=0.15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0.5,1.0])
        ax.set_ylim([-0.2,0.2])
        ax.set_zlim([0.0,0.3])
        ax.set_title(titles[i])
    #plt.show()
    plt.savefig(title)
    plt.clf()
    plt.cla()
    plt.close(fig)

def downsample_pointcloud(xyz, n_points):
    random_idxs = np.random.choice(np.arange(len(xyz)), n_points)
    downsampled_xyz = xyz[random_idxs]
    return downsampled_xyz

def filter_pointcloud(xyz):
    p = pcl.PointCloud(xyz.astype(np.float32))
    clipper = p.make_cropbox()
    tx = 0
    ty = 0
    tz = 0
    clipper.set_Translation(tx, ty, tz)
    rx = 0
    ry = 0
    rz = 0
    clipper.set_Rotation(rx, ry, rz)
    minx = 0.5
    miny = -0.2
    minz = 0.075
    mins = 0
    maxx = 0.9
    maxy = 0.2
    maxz = 1.0
    maxs = 0
    clipper.set_MinMax(minx, miny, minz, mins, maxx, maxy, maxz, maxs)
    p = clipper.filter()

    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k(175)
    fil.set_std_dev_mul_thresh(0.5)
    result = fil.filter().to_array()
    return result

def inpaint_depth(pixel_xyz):
    depth = pixel_xyz[:,:,2]
    idxs = np.where(np.around(depth,3) == 0.728)
    mask = np.ones_like(depth)
    mask[idxs] = 0.0
    return idxs

def process(mask_dir, depth_dir, idx, n_points):
    mask = cv2.imread(os.path.join(mask_dir, 'mask_%d.jpg'%idx), 0)
    _, mask = cv2.threshold(mask, 127,255,cv2.THRESH_BINARY)
    pixel_xyz = np.load(os.path.join(depth_dir, 'pixel_xyz_%d.npz'%idx))['arr_0'][:,80:560]

    #3idxs = inpaint_depth(pixel_xyz)
    #3mask[idxs] = 255

    points_unmasked  = pixel_xyz[mask == 0]
    #points_unmasked = points_unmasked[points_unmasked[:,2] > 0.075]
    #points_unmasked = points_unmasked[points_unmasked[:,2] > 0.09]
    #points_unmasked = points_unmasked[points_unmasked[:,0] < 0.95]

    #pcl = downsample_pointcloud(points_unmasked, n_points)
    pcl = points_unmasked
    pcl = filter_pointcloud(pcl)
    return pcl

def main(overhead_mask_dir, overhead_depth_dir, side_mask_dir, side_depth_dir, output_dir, n_points):
    for i in range(len(os.listdir(overhead_mask_dir))):
        print(i)
        pcl_overhead = process(overhead_mask_dir, overhead_depth_dir, i, n_points)
        pcl_side = process(side_mask_dir, side_depth_dir, i, n_points)
        pcl_side += np.array([0.08,0.1,0.1])
        pcl = np.vstack((pcl_overhead, pcl_side))
        #pcl = pcl_side
        #pcl = pcl_overhead
        #plot_pointclouds([pcl], title='%s/%03d.jpg'%(output_dir, i))
        #plot_pointclouds([pcl_side, pcl_overhead], title='%s/%03d.jpg'%(output_dir, i))
        plot_pointclouds([pcl], title='%s/%03d.jpg'%(output_dir, i))
        np.save('%s/%03d.npy'%(output_dir, i), pcl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--rollout_dir', type=str)
    parser.add_argument('-n', '--num_points', type=int, default=10000)
    args = parser.parse_args()
    overhead_mask_dir = os.path.join(args.rollout_dir, 'overhead', 'mask_dilated')
    overhead_depth_dir = os.path.join(args.rollout_dir, 'overhead', 'depth')
    side_mask_dir = os.path.join(args.rollout_dir, 'side', 'mask_dilated')
    side_depth_dir = os.path.join(args.rollout_dir, 'side', 'depth')
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    main(overhead_mask_dir, overhead_depth_dir, side_mask_dir, side_depth_dir, output_dir, args.num_points)

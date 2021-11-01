import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import cv2
import argparse

def plot_pointclouds(pcls, title=""):
    fig = plt.figure(figsize=(10, 5))
    titles=['curr', 'ref']
    for i,points in enumerate(pcls):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        x, y, z = points.T
        ax.scatter3D(x, y, z, s=0.15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0.3,1.2])
        ax.set_ylim([-0.2,0.2])
        ax.set_zlim([0,0.4])
        ax.set_title(titles[i])
    plt.savefig(title)
    plt.clf()
    plt.cla()
    plt.close(fig)

def downsample_pointcloud(xyz, n_points):
    random_idxs = np.random.choice(np.arange(len(xyz)), n_points)
    downsampled_xyz = xyz[random_idxs]
    return downsampled_xyz

def main(mask_dir, depth_dir, output_dir, n_points, traj):
    for i in range(len(os.listdir(mask_dir))):
        mask = cv2.imread(os.path.join(mask_dir, 'mask_%d.jpg'%i), 0)
        _, mask = cv2.threshold(mask, 127,255,cv2.THRESH_BINARY)
        pixel_xyz = np.load(os.path.join(depth_dir, 'pixel_xyz_%d.npz'%i))['arr_0'][:,80:560]
        points_unmasked  = pixel_xyz[mask == 0]
        points_unmasked = points_unmasked[points_unmasked[:,2] > 0.075]
        points_unmasked = points_unmasked[points_unmasked[:,0] < 0.95]
        points_unmasked += np.array([-0.03, 0, 0])
        pcl = downsample_pointcloud(points_unmasked, n_points)
        plot_pointclouds([pcl], title='%s/%03d.jpg'%(output_dir, i))
        np.save('%s/%03d.npy'%(output_dir, i), pcl)
        print(mask.shape, pixel_xyz.shape, points_unmasked.shape)
    np.save('%s/traj.npy'%(output_dir), traj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--rollout_dir', type=str)
    parser.add_argument('-n', '--num_points', type=int, default=10000)
    args = parser.parse_args()
    path_to_traj = os.path.join(args.rollout_dir, 'traj_for_imgs.npz')
    traj = np.load(path_to_traj)['pos_traj']
    mask_dir = os.path.join(args.rollout_dir, 'mask_dilated')
    depth_dir = os.path.join(args.rollout_dir, 'depth')
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    main(mask_dir, depth_dir, output_dir, args.num_points, traj)

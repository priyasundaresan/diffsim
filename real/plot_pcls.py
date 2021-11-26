import matplotlib.pyplot as plt
import pprint
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import json

def plot_pointclouds(pcls, title="", angle=0):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i,points in enumerate(pcls):
        x, y, z = points.T
        ax.scatter3D(x, y, z, s=0.2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # this is for sim frame
        ax.set_xlim([-3,3])
        ax.set_ylim([-3,3])
        ax.set_zlim([0,2.0])

        #ax.set_xlim([-0.6,0.6])
        #ax.set_ylim([-0.6,0.6])
        #ax.set_zlim([0,0.75])

        # this is for real frame
        #ax.set_xlim([0.4,1.0])
        #ax.set_ylim([-0.4,0.4])
        #ax.set_zlim([0,0.4])

        ax.view_init(10, angle)

    #plt.show()
    plt.savefig(title)
    plt.clf()
    plt.cla()
    plt.close(fig)

def traj_to_motion(traj, frame_time, skip=2):
    data = {"motions": [{"time": 0}]}
    #for i in range(1,len(traj),skip):
    for i in range(len(traj),skip):
        waypt = traj[i]
        val_i = {"time": (i+1)*frame_time, "transform": {"translate": list(waypt), "scale": 0}}
        data["motions"].append(val_i)
    data["motions"] = [data["motions"]]
    with open('motion.json', 'w') as f:
        json.dump(data, f, indent=4)
    pprint.pprint(data)

def traj_to_vel(traj):
    traj_t1 = traj[1:]
    dt = 0.01
    vels = (traj_t1 - traj[:-1])/dt
    np.save('vels.npy', vels)
    #print(vels)

def real2sim_transf(points, trans, scale, rot_inv):
    return ((points - trans)/scale)@rot_inv

def sim2real_transf(points, trans, scale, rot):
    return points@rot*scale + trans

if __name__ == '__main__':
    task = 'stretch'
    #task = 'fling'
    real_dir = 'output'
    #sim_dir = 'fling_fast_drop_pcl'
    sim_dir = 'stretch_pcl'

    output_dir = 'pcls'
    if not (os.path.exists(output_dir)):
        os.mkdir(output_dir)
        
    # fling
    if task == 'fling':
        rot = R.from_euler('z', 90, degrees=True).as_matrix()
        rot_inv = R.from_euler('z', 90, degrees=True).inv().as_matrix()
        trans = np.array([0.75,0.02,-0.05])
        scale = 0.35
    elif task == 'stretch':
        rot = R.from_euler('z', -90, degrees=True).as_matrix()
        rot_inv = R.from_euler('z', -90, degrees=True).inv().as_matrix()
        #trans = np.array([0.65,0.04,0.0385])
        trans = np.array([0.68,0.04,0.0385])
        scale = 0.12

    # transforming real --> sim frame
    #sim_traj = np.load('fling_fast_sim_traj.npy')
    #real_traj = sim2real_transf(sim_traj, trans, scale, rot_inv)
    #print(np.around(real_traj,2))
    #print(np.around(np.load('fling_real_pos.npy'), 2))

    episode_length = len(os.listdir(real_dir))//2
    # loop
    for i in range(episode_length):
        pcl1 = np.load(os.path.join(real_dir, '%03d.npy'%i))
        pcl2 = np.load(os.path.join(sim_dir, '%03d.npy'%i)).squeeze()

        # transforming real --> sim frame
        #pcl1 = ((pcl1 - np.array([0.75,0.02,-0.05]))/0.35)@rot_inv # fling
        #pcl1 = ((pcl1 - np.array([0.65,0.04,0.0385]))/0.12)@rot_inv # loop

        pcl1 = real2sim_transf(pcl1, trans, scale, rot)

        plot_pointclouds([pcl1, pcl2], title=os.path.join(output_dir, '%03d.jpg'%i), angle=(i*360/episode_length))
        print(i)

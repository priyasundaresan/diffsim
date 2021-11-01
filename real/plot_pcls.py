import matplotlib.pyplot as plt
import pprint
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import json

def plot_pointclouds(pcls, title=""):
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

        # this is for real frame
        #ax.set_xlim([0.4,1.0])
        #ax.set_ylim([-0.4,0.4])
        #ax.set_zlim([0,0.4])

        ax.view_init(30, 0)

    #plt.show()
    plt.savefig(title)
    plt.clf()
    plt.cla()
    plt.close(fig)

def traj_to_motion(traj, frame_time, skip=2):
    end_time = int(len(traj)*frame_time)
    #print(end_time)
    data = {"motions": [{"time": 0}]}
    for i in range(1,len(traj),skip):
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

if __name__ == '__main__':
    real_dir = 'output'
    sim_dir = os.path.abspath(os.path.join('..', 'pysim', 'demo_pcl_frames'))

    output_dir = 'pcls'
    traj = np.load(os.path.join(real_dir, 'traj.npy'))
    r1 = R.from_euler('z', -90, degrees=True).as_matrix()
    r1_inv = R.from_euler('z', -90, degrees=True).inv().as_matrix()
    if not (os.path.exists(output_dir)):
        os.mkdir(output_dir)
    # transforming real --> sim frame
    transf_traj = ((traj - np.array([0.75,0,0]))/0.2)@r1_inv

    traj_to_motion(transf_traj, 0.04)
    traj_to_vel(transf_traj)

    #sim_pos = np.load('fling_sim_pos.npy')
    #real_pos = (sim_pos@r1)*0.2 + np.array([0.75,0,0])
    #np.save('fling_real_pos.npy', real_pos)
    #print(np.around(traj, 2))
    #print(np.around(real_pos, 2))
        
    for i in range(len(os.listdir(sim_dir))):
        pcl1 = np.load(os.path.join(real_dir, '%03d.npy'%i))
        pcl2 = np.load(os.path.join(sim_dir, '%03d.npy'%i)).squeeze()

        # transforming real --> sim frame
        pcl1 = ((pcl1 - np.array([0.75,0,0]))/0.2)@r1_inv

        # transforming sim --> real frame
        #pcl2 = (pcl2@r1)*0.2 + np.array([0.75,0,0])
        #traj = (traj@r1)*0.2 + np.array([0.75,0,0])
        #plot_pointclouds([pcl1, pcl2, traj], title=os.path.join(output_dir, '%03d.jpg'%i))

        plot_pointclouds([pcl1, pcl2, transf_traj], title=os.path.join(output_dir, '%03d.jpg'%i))
        print(i)

import os
import sys

import torch
import pprint
import torch.nn as nn
import torch.nn.functional as F
import arcsim
import gc
import time
import json
import gc
import numpy as np
from datetime import datetime

import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.io import load_obj

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
estimate 11oz-black-denim.json starting from gray-interlock.json
'''

if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

with open('conf/rigidcloth/material_est/start.json','r') as f:
	config = json.load(f)

# PYTORCH3D
device = torch.device("cuda:0")

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')

torch.set_num_threads(8)
spf = config['frame_steps']
total_steps = 40
scalev=1
num_points = 5000


def reset_sim(sim, epoch):
	arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)

def get_render_mesh_from_sim(sim):
    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)

    all_verts = [cloth_verts]
    all_faces = [cloth_faces]

    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)])
    return mesh

def plot_pointclouds(pcls, title=""):
    fig = plt.figure(figsize=(10, 5))
    titles=['curr', 'ref']
    for i,points in enumerate(pcls):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        x, z, y = points.clone().detach().cpu().squeeze().unbind(1)    
        ax.scatter3D(x, z, y, s=0.15)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_xlim([-1,1])
        ax.set_ylim([-0.5,1.0])
        ax.set_zlim([-1,0.5])
        ax.set_title(titles[i])
        ax.view_init(30, -50)
    plt.savefig(title)
    plt.clf()
    plt.cla()
    plt.close(fig)

def get_loss_per_iter(sim, epoch, sim_step):
    curr_mesh = get_render_mesh_from_sim(sim)
    curr_pcl = sample_points_from_meshes(curr_mesh, num_points)
    ref_pcl = torch.from_numpy(np.load('demo_exp_estimate_material/%03d.npy'%sim_step)).to(device)
    loss_chamfer, _ = chamfer_distance(ref_pcl, curr_pcl)
    if epoch % 5 == 0:
        plot_pointclouds([curr_pcl, ref_pcl], title='%s/epoch%02d-%03d'%(out_path,epoch,sim_step))
    return loss_chamfer

def run_sim(steps, sim, epoch):
    sim.cloths[0].materials[0].stretchingori = param_stretch
    sim.cloths[0].materials[0].bendingori = param_bend
    pprint.pprint(param_stretch)
    pprint.pprint(param_bend)
    loss = 0.0
    for step in range(steps):
        print(step)
        arcsim.sim_step()
        loss += get_loss_per_iter(sim, epoch, step)
    
    loss /= steps
    reg  = torch.norm(param_stretch, p=2)*0.00001 + torch.norm(param_bend, p=2)*0.001
    return loss.cpu()+reg, loss.cpu()

def do_train(cur_step,optimizers,sim):
    epoch = 0
    while True:
        steps = total_steps
        
        reset_sim(sim, epoch)
        
        st = time.time()
        loss, loss_chamfer = run_sim(steps, sim, epoch)
        en0 = time.time()
        
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        loss.backward()
        
        en1 = time.time()
        print("=======================================")
        print('epoch {}: loss={} loss_chamfer={} \n'.format(epoch, loss.data, loss_chamfer.data))
        
        print('forward tim = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))

        if loss < 0.001:
            break
        
        for optimizer in optimizers:
            optimizer.step()
        epoch = epoch + 1
    return loss
# break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
    tot_step = 1
    sim=arcsim.get_sim()
    
    out_dir = 'exps_stiff_grav'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    results = []
    cur_step = 0
    reset_sim(sim, 0)
    param_stretch = sim.cloths[0].materials[0].stretchingori
    param_bend = sim.cloths[0].materials[0].bendingori
    optimizer_stretch = torch.optim.Adam([param_stretch],lr=10)
    optimizer_bend = torch.optim.Adam([param_stretch],lr=1e-7)
    loss = do_train(0,[optimizer_stretch, optimizer_bend],sim)
                                 
print("done")

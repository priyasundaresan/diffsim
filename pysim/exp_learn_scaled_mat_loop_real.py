import torch
import pprint
import torch.nn as nn
import torch.nn.functional as F
import arcsim
import gc
import time
import json
import sys
import gc
import numpy as np
import os
from datetime import datetime

import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.transform import Rotation as R
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

from load_material_props import load_material, combine_materials
from utils.chamfer import chamfer_distance_one_sided

device = torch.device("cuda:0")

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def stretch_fn(x):
    #return 35.3*(x**3) - 21.03*(x**2) + 5.354*x + 0.5278
    return 37.61574074*(x**3) + 3.17460317*(x**2) - 1.48478836*x + 0.54365079

def bend_fn(x):
    return 22.84*(x**3) - 8.358*(x**2) + 5.415*x + 0.5505

#with open('conf/rigidcloth/loop_stretch/demo2.json','r') as f:
with open('conf/rigidcloth/loop_stretch/demo3.json','r') as f:
	config = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')

torch.set_num_threads(8)
spf = config['frame_steps']
total_steps = 30
num_points = 10000

scalev=1

rotation_pcl = torch.from_numpy(R.from_euler('z', -90, degrees=True).as_matrix()).to(device).float()
translation_pcl = torch.Tensor([0.75,0,0]).to(device)

def reset_sim(sim, epoch):
    arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)


def plot_pointclouds(pcls, title=""):
    fig = plt.figure(figsize=(30, 5))
    titles=['curr', 'ref']

    ax_both = fig.add_subplot(141, projection='3d')
    ax_curr = fig.add_subplot(143, projection='3d')
    ax_ref = fig.add_subplot(144, projection='3d')
    ax_initial = fig.add_subplot(142, projection='3d')
    colors = ['deepskyblue', 'green', 'black']
    labels = ['curr', 'ref', 'initial']
    for i,(color,label,points) in enumerate(zip(colors,labels,pcls)):
        x, y, z = points.detach().cpu().numpy().T
        #label = 'curr' if i==0 else 'ref'
        #color = '#17becf' if i==0 else 'red'
        ax_both.scatter3D(x, y, z, s=0.2, label=label, color=color)
        if i==0:
            ax_curr.scatter3D(x, y, z, s=0.2, label=label, color=color)
        elif i==1:
            ax_ref.scatter3D(x, y, z, s=0.2, label=label, color=color)
        else:
            ax_initial.scatter3D(x, y, z, s=0.2, label=label, color=color)
        ax_both.set_xlabel('x')
        ax_both.set_ylabel('y')
        ax_both.set_zlabel('z')

        # this is for sim frame
        ax_both.set_xlim([-1,1])
        ax_both.set_ylim([-1,1])
        ax_both.set_zlim([0,1.3])

        ax_curr.set_xlim([-1,1])
        ax_curr.set_ylim([-1,1])
        ax_curr.set_zlim([0,1.3])

        ax_ref.set_xlim([-1,1])
        ax_ref.set_ylim([-1,1])
        ax_ref.set_zlim([0,1.3])

        ax_initial.set_xlim([-1,1])
        ax_initial.set_ylim([-1,1])
        ax_initial.set_zlim([0,1.3])

        ax_both.view_init(30, 40)
        ax_curr.view_init(30, 40)
        ax_ref.view_init(30, 40)
        ax_initial.view_init(30, 40)

    ax_both.legend()
    ax_curr.legend()
    ax_ref.legend()
    ax_initial.legend()

    plt.savefig(title)
    plt.clf()
    plt.cla()
    plt.close(fig)

def get_render_mesh_from_sim(sim):
    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)

    pole_verts = torch.stack([v.node.x for v in sim.obstacles[0].curr_state_mesh.verts]).float().to(device)
    pole_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.obstacles[0].curr_state_mesh.faces]).to(device)
    pole_faces += len(cloth_verts)

    all_verts = [cloth_verts, pole_verts]
    all_faces = [cloth_faces, pole_faces]

    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)])
    return mesh

def get_cloth_mesh_from_sim(sim):
    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)
    all_verts = [cloth_verts]
    all_faces = [cloth_faces]
    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)])
    return mesh

def real2sim_transf(points, trans, scale, rot_inv):
    return ((points - trans)/scale)@rot_inv

#def get_ref_pcl(sim_step, demo_dir='demo_exp_learn_scaled_mat_loop_real'):
#def get_ref_pcl(sim_step, demo_dir='demo_exp_learn_scaled_mat_loop_real_stretchblack3'):
def get_ref_pcl(sim_step, demo_dir='demo_exp_learn_scaled_mat_loop_real_stretchblack2'):
    rot = R.from_euler('z', 90, degrees=True).as_matrix()
    rot_inv = R.from_euler('z', 90, degrees=True).inv().as_matrix()
    trans = np.array([0.65,0.04,0.0385])
    scale = 0.12
    ref_pcl = np.load(os.path.join(demo_dir, '%03d.npy'%sim_step))
    ref_pcl = real2sim_transf(ref_pcl, trans, scale, rot_inv)
    ref_pcl = torch.from_numpy(ref_pcl).to(device).unsqueeze(0).float()
    return ref_pcl

def get_loss_per_iter(sim, epoch, sim_step, save, initial_states=None):
    curr_mesh = get_render_mesh_from_sim(sim)
    curr_mesh_cloth_only = get_cloth_mesh_from_sim(sim)
    curr_pcl = sample_points_from_meshes(curr_mesh, num_points)
    curr_pcl_cloth_only = sample_points_from_meshes(curr_mesh_cloth_only, num_points)
    #ref_mesh = get_ref_mesh(sim_step)
    #ref_pcl = sample_points_from_meshes(ref_mesh, num_points)
    ref_pcl = get_ref_pcl(sim_step)

    #loss_chamfer, _ = chamfer_distance(curr_pcl, ref_pcl)
    loss_chamfer, _ = chamfer_distance_one_sided(curr_pcl_cloth_only, ref_pcl)
    if (save):
        initial_mesh = initial_states[sim_step]
        initial_pcl = sample_points_from_meshes(initial_mesh, num_points)
        plot_pointclouds([curr_pcl, ref_pcl, initial_pcl], title='%s/epoch%02d-%03d'%(out_path,epoch,sim_step))
    return loss_chamfer, curr_mesh

def run_sim(steps, sim, epoch, save, initial_states=None):
    bend_multiplier, stretch_multiplier = torch.sigmoid(param_g)
    loss = 0.0

    orig_stretch = sim.cloths[0].materials[0].stretching
    orig_bend = sim.cloths[0].materials[0].bending
    orig_density = sim.cloths[0].materials[0].densityori

    new_stretch_multiplier = stretch_fn(stretch_multiplier)
    new_bend_multiplier = bend_fn(bend_multiplier)
    sim.cloths[0].materials[0].stretching = orig_stretch*new_stretch_multiplier
    sim.cloths[0].materials[0].bending = orig_bend*new_bend_multiplier

    print("stretch, bend", (new_stretch_multiplier, new_bend_multiplier))
    print(param_g.grad)

    mesh_states = []
    updates = 0
    for step in range(15):
        print(step)
        arcsim.sim_step()
        loss_curr, curr_mesh = get_loss_per_iter(sim, epoch, step, save=save, initial_states=initial_states)
        mesh_states.append(curr_mesh)
    for step in range(15,steps):
        print(step)
        arcsim.sim_step()
        loss_curr, curr_mesh = get_loss_per_iter(sim, epoch, step, save=save, initial_states=initial_states)
        loss += loss_curr
        updates += 1
        mesh_states.append(curr_mesh)

    loss /= updates

    return loss, mesh_states

def do_train(cur_step,optimizer,sim,initial_states):
    epoch = 0
    loss = float('inf')
    thresh = 0.012
    num_steps_to_run = 16
    while True:
        
        reset_sim(sim, epoch)
        
        st = time.time()
        loss,_ = run_sim(num_steps_to_run, sim, epoch, save=(epoch%10==0), initial_states=initial_states)
        if epoch > 100:
            print('epic fail')
            break

        if num_steps_to_run == total_steps:
            break

        if loss < thresh:
            num_steps_to_run += 1

        #if loss < thresh:
        #    break

        en0 = time.time()
        optimizer.zero_grad()
        loss.backward()
        en1 = time.time()
        if loss >= thresh:
            optimizer.step()
        print("=======================================")
        print('epoch {}: loss={}\n'.format(epoch, loss.data))
        
        print('forward tim = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))
       
        epoch = epoch + 1
        # break
    return F.softmax(param_g), loss, epoch

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
    tot_step = 1
    sim=arcsim.get_sim()
    
    initial_probs = torch.tensor([0.5,0.5])
    #initial_probs = torch.tensor([0.095,0.17])
    param_g = torch.log(initial_probs/(torch.ones_like(initial_probs)-initial_probs))
    print("here", torch.sigmoid(param_g))
    param_g.requires_grad = True
    lr = 0.3
    optimizer = torch.optim.Adam([param_g],lr=lr)
    reset_sim(sim, 0)
    _, initial_states = run_sim(total_steps, sim, 0, save=False)
    result,loss,iters = do_train(tot_step,optimizer,sim,initial_states)
    reset_sim(sim, iters+1)
    run_sim(total_steps, sim, iters+1, save=True, initial_states=initial_states)
    
print("done")


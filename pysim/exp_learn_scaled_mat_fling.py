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

device = torch.device("cuda:0")

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

#stretch_subspace = torch.Tensor([0.5,1,2,3,10,20]).to(device)
#stretch_idxs = torch.linspace(0,1,len(stretch_subspace)).to(device)
#bend_subspace = torch.Tensor([0.5,1,2,3,4,5,10,15,20]).to(device)
#bend_idxs = torch.linspace(0,1,len(bend_subspace)).to(device)
#stretch_fn = NaturalCubicSpline(natural_cubic_spline_coeffs(stretch_idxs, stretch_subspace.unsqueeze(1)))
#bend_fn = NaturalCubicSpline(natural_cubic_spline_coeffs(bend_idxs, bend_subspace.unsqueeze(1)))

def stretch_fn(x):
    return 35.3*(x**3) - 21.03*(x**2) + 5.354*x + 0.5278
def bend_fn(x):
    return 22.84*(x**3) - 8.358*(x**2) + 5.415*x + 0.5505

with open('conf/rigidcloth/fling/demo_shorter_cloth_camelponteroma.json','r') as f:
	config = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')

torch.set_num_threads(8)
spf = config['frame_steps']
total_steps = 40
num_points = 10000

scalev=1

rotation_pcl = torch.from_numpy(R.from_euler('z', -90, degrees=True).as_matrix()).to(device).float()
translation_pcl = torch.Tensor([0.75,0,0]).to(device)

def reset_sim(sim, epoch):
    arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)


def plot_pointclouds(pcls, title=""):
    fig = plt.figure(figsize=(10, 5))
    titles=['curr', 'ref']

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i,points in enumerate(pcls):
        #ax = fig.add_subplot(1, 2, i+1, projection='3d')
        x, y, z = points.detach().cpu().numpy().T
        label = 'curr' if i==0 else 'ref'
        ax.scatter3D(x, y, z, s=0.2, label=label)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # this is for sim frame
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([0,1.3])

        # this is for real frame
        #ax.set_xlim([0.5,0.9])
        #ax.set_ylim([-0.4,0.4])
        #ax.set_zlim([0,0.39])
        #ax.view_init(30, 85)
        ax.view_init(30, 40)
    ax.legend()
    plt.savefig(title)
    plt.clf()
    plt.cla()
    plt.close(fig)

def get_render_mesh_from_sim(sim):
    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)

    all_verts = [cloth_verts]
    all_faces = [cloth_faces]

    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)])
    return mesh

def get_ref_mesh(sim_step):
    demo_dir = 'demo_exp_learn_scaled_mat_fling_ivoryribknit'
    mesh_fnames = sorted([f for f in os.listdir('%s/out0'%(demo_dir)) if '%04d'%sim_step in f])
    all_verts = []
    all_faces = []
    all_textures = []
    vert_count = 0
    for j, f in enumerate(mesh_fnames[:1]):
        verts, faces, aux = load_obj(os.path.join(demo_dir, "out0", f))
        faces_idx = faces.verts_idx.to(device) + vert_count
        verts = verts.to(device)
        vert_count += len(verts)
        all_verts.append(verts)
        all_faces.append(faces_idx)
    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)])
    return mesh

def get_loss_per_iter(sim, epoch, sim_step, save):
    demo_dir = 'demo_exp_learn_mat_fling_real_pcl_video'
    curr_mesh = get_render_mesh_from_sim(sim)
    curr_pcl = sample_points_from_meshes(curr_mesh, num_points)
    #transformed_curr_pcl = (curr_pcl@rotation_pcl)*0.2 + translation_pcl
    ref_mesh = get_ref_mesh(sim_step)
    ref_pcl = sample_points_from_meshes(ref_mesh, num_points)
    #ref_pcl = (ref_pcl@rotation_pcl)*0.2 + translation_pcl
    #ref_pcl = torch.from_numpy(np.load('%s/%03d.npy'%(demo_dir, sim_step))).to(device).unsqueeze(0).float()
    #loss_chamfer, _ = chamfer_distance(transformed_curr_pcl, ref_pcl)
    loss_chamfer, _ = chamfer_distance(curr_pcl, ref_pcl)
    if save:
        #plot_pointclouds([transformed_curr_pcl, ref_pcl], title='%s/epoch%02d-%03d'%(out_path,epoch,sim_step))
        plot_pointclouds([curr_pcl, ref_pcl], title='%s/epoch%02d-%03d'%(out_path,epoch,sim_step))
    return loss_chamfer

def run_sim(steps, sim, epoch, save):
    bend_multiplier, stretch_multiplier = torch.sigmoid(param_g)
    loss = 0.0

    orig_stretch = sim.cloths[0].materials[0].stretching
    orig_bend = sim.cloths[0].materials[0].bending

    #new_stretch_multiplier = stretch_multiplier*20
    #new_bend_multiplier = bend_multiplier*20

    new_stretch_multiplier = stretch_fn(stretch_multiplier)
    new_bend_multiplier = bend_fn(bend_multiplier)
    sim.cloths[0].materials[0].stretching = orig_stretch*new_stretch_multiplier
    sim.cloths[0].materials[0].bending = orig_bend*new_bend_multiplier

    print("stretch and bend:", (stretch_multiplier, new_stretch_multiplier), (bend_multiplier, new_bend_multiplier))
    print(param_g.grad)

    for step in range(steps):
        print(step)
        arcsim.sim_step()
        loss += get_loss_per_iter(sim, epoch, step, save=save)
    loss /= steps

    #reg  = torch.norm(param_g, p=2)*0.001 
    #return loss + reg.cuda()
    return loss

def do_train(cur_step,optimizer,sim):
    epoch = 0
    loss = float('inf')
    thresh = 0.007
    #num_steps_to_run = total_steps
    num_steps_to_run = 1
    while True:
        
        reset_sim(sim, epoch)
        
        st = time.time()
        loss = run_sim(num_steps_to_run, sim, epoch, save=(epoch%5==0))
        if epoch > 150:
            print('epic fail')
            break

        if num_steps_to_run == total_steps:
            break

        if loss < thresh:
            num_steps_to_run += 1

        en0 = time.time()
        optimizer.zero_grad()
        loss.backward()
        en1 = time.time()
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
    param_g = torch.log(initial_probs/(torch.ones_like(initial_probs)-initial_probs))
    print("here", torch.sigmoid(param_g))
    param_g.requires_grad = True
    #lr = 0.1
    lr = 0.2
    optimizer = torch.optim.Adam([param_g],lr=lr)
    result,loss,iters = do_train(tot_step,optimizer,sim)
    reset_sim(sim, iters+1)
    run_sim(total_steps, sim, iters+1, save=True)
    
print("done")


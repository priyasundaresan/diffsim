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

from load_material_props import load_material, combine_materials
materials = ['11oz-black-denim.json', 'gray-interlock.json', 'navy-sparkle-sweat.json']
#materials = ['11oz-black-denim.json', 'gray-interlock.json', 'ivory-rib-knit.json']
#materials = ['11oz-black-denim.json', 'gray-interlock.json']
base_dir = 'materials'
density_all = []
bending_all = []
stretching_all = []
for m in materials:
    d,b,s = load_material(os.path.join(base_dir, m), torch.device("cuda:0")) 
    density_all.append(d)
    bending_all.append(b.tolist())
    stretching_all.append(s.tolist())
density_all = torch.Tensor(density_all)
bending_all = torch.Tensor(bending_all)
stretching_all = torch.Tensor(stretching_all)

device = torch.device("cuda:0")

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


with open('conf/rigidcloth/lasso/demo_slow.json','r') as f:
	config = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')

torch.set_num_threads(8)
spf = config['frame_steps']
total_steps = 50
num_points = 5000

scalev=1


def reset_sim(sim, epoch):
    arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)

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
        ax.view_init(30,-75)
    plt.savefig(title)
    plt.clf()
    plt.cla()
    plt.close(fig)

def get_render_mesh_from_sim(sim):
    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)

    pole_verts = torch.stack([v.node.x for v in sim.obstacles[1].curr_state_mesh.verts]).float().to(device)
    pole_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.obstacles[1].curr_state_mesh.faces]).to(device)
    pole_faces += len(cloth_verts)

    all_verts = [cloth_verts, pole_verts]
    all_faces = [cloth_faces, pole_faces]

    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)])
    return mesh

def get_ref_mesh(sim_step):
    mesh_fnames = sorted([f for f in os.listdir('demo_exp_learn_lasso_materialest/out0') if '%04d'%sim_step in f])
    all_verts = []
    all_faces = []
    all_textures = []
    vert_count = 0
    for j, f in enumerate(mesh_fnames[:1] + mesh_fnames[2:]):
        verts, faces, aux = load_obj(os.path.join("demo_exp_learn_lasso_materialest", "out0", f))
        faces_idx = faces.verts_idx.to(device) + vert_count
        verts = verts.to(device)
        vert_count += len(verts)
        all_verts.append(verts)
        all_faces.append(faces_idx)
    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)])
    return mesh

def get_loss_per_iter(sim, epoch, sim_step):
    curr_mesh = get_render_mesh_from_sim(sim)
    curr_pcl = sample_points_from_meshes(curr_mesh, num_points)
    ref_mesh = get_ref_mesh(sim_step)
    ref_pcl = sample_points_from_meshes(ref_mesh, num_points)
    loss_chamfer, _ = chamfer_distance(ref_pcl, curr_pcl)
    if epoch % 10 == 0:
        plot_pointclouds([curr_pcl, ref_pcl], title='%s/epoch%02d-%03d'%(out_path,epoch,sim_step))
    return loss_chamfer

def run_sim(steps, sim, epoch):
    #reg  = torch.norm(param_g, p=2)*0.001
    loss = 0.0
    proportions = F.softmax(param_g).float()
    print("proportions", proportions)
    density, bend, stretch = combine_materials(density_all, bending_all, stretching_all, proportions)
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim.cloths[0].materials[0].densityori= density
    sim.cloths[0].materials[0].stretchingori = stretch
    sim.cloths[0].materials[0].bendingori = bend
    arcsim.reload_material_data(sim.cloths[0].materials[0])
    for step in range(total_steps):
        print(step)
        arcsim.sim_step()
        loss += get_loss_per_iter(sim, epoch, step)
    loss /= steps
    #return loss + reg.cuda()
    return loss

def do_train(cur_step,optimizer,sim):
    epoch = 0
    loss = float('inf')
    thresh = 0.001
    num_steps_to_run = total_steps
    while True:
        
        reset_sim(sim, epoch)
        
        st = time.time()
        loss = run_sim(num_steps_to_run, sim, epoch)
        
        if loss < thresh:
            print(loss)
            break
        if epoch > 10:
            loss = float('inf')
            break

        en0 = time.time()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        en1 = time.time()
        print("=======================================")
        print('epoch {}: loss={}\n'.format(epoch, loss.data))
        
        print('forward tim = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))
        
        #if epoch % 5 == 0:
        #	torch.save(net.state_dict(), torch_model_path)
        
        optimizer.step()
        epoch = epoch + 1
        # break
    return F.softmax(param_g), loss, epoch

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
    tot_step = 1
    sim=arcsim.get_sim()
    
    initial_probs = torch.tensor([0.33,0.33,0.33],dtype=torch.float64)
    param_g = torch.log(initial_probs)
    param_g.requires_grad = True
    lr = 0.2
    optimizer = torch.optim.Adam([param_g],lr=lr)
    for cur_step in range(tot_step):
        do_train(cur_step,optimizer,sim)
    
    out_dir = 'exps_lasso_materialest'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    #results = []
    #cur_step = 0
    #for i in np.linspace(0.2,0.4,5):
    #    for j in np.linspace(0.2,0.4,5):
    #        k = 1.0 - i - j
    #        pprint.pprint(results)
    #        initial_probs = torch.tensor([i,j,k])
    #        param_g = torch.log(initial_probs)
    #        print(initial_probs, F.softmax(param_g))
    #        param_g.requires_grad = True
    #        lr = 0.2
    #        optimizer = torch.optim.Adam([param_g],lr=lr)
    #        result, loss, iters = do_train(cur_step,optimizer,sim)
    #        if loss != float('inf'):
    #            results.append([i,j,k] + result.squeeze().tolist() + [loss.item()] + [iters])
    #        os.system('mv %s ./%s/run%d'%(out_path, out_dir,cur_step))
    #        os.system('mkdir %s'%out_path)
    #        save_config(config, out_path+'/conf.json')
    #        cur_step += 1
    #        np.save('%s/results.npy'%out_dir, results)
    #pprint.pprint(results)
    #results = np.array(results)
    #np.save('%s/results.npy'%out_dir, results)
                                 
print("done")


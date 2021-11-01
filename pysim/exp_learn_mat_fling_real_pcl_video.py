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

from load_material_props import load_material, combine_materials
materials = ['white-swim-solid.json', 'camel-ponte-roma.json']
#materials = ['11oz-black-denim.json', 'gray-interlock.json', 'navy-sparkle-sweat.json', 'paper.json']
#materials = ['gray-interlock.json', 'navy-sparkle-sweat.json', 'paper.json']
#materials = ['11oz-black-denim.json', \
#             'camel-ponte-roma.json', \
#             'gray-interlock.json', \
#             'ivory-rib-knit.json', \
#             'navy-sparkle-sweat.json', \
#             'paper.json', \
#             'pink-ribbon-brown.json', \
#             'royal-target.json', \
#             'tango-red-jet-set.json', \
#             'white-dots-on-blk.json', \
#             'white-swim-solid.json']

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


with open('conf/rigidcloth/fling/demo_shorter_cloth.json','r') as f:
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

    #ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i,points in enumerate(pcls):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        x, y, z = points.detach().cpu().numpy().T
        ax.scatter3D(x, y, z, s=0.2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # this is for real frame
        ax.set_xlim([0.5,0.9])
        ax.set_ylim([-0.4,0.4])
        ax.set_zlim([0,0.39])
        ax.view_init(30, 85)

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


def get_loss_per_iter(sim, epoch, sim_step, save):
    demo_dir = 'demo_exp_learn_mat_fling_real_pcl_video'
    curr_mesh = get_render_mesh_from_sim(sim)
    curr_pcl = sample_points_from_meshes(curr_mesh, num_points)
    transformed_curr_pcl = (curr_pcl@rotation_pcl)*0.2 + translation_pcl
    ref_pcl = torch.from_numpy(np.load('%s/%03d.npy'%(demo_dir, sim_step))).to(device).unsqueeze(0).float()
    loss_chamfer, _ = chamfer_distance(transformed_curr_pcl, ref_pcl)
    if save:
        plot_pointclouds([transformed_curr_pcl, ref_pcl], title='%s/epoch%02d-%03d'%(out_path,epoch,sim_step))
    return loss_chamfer

def run_sim(steps, sim, epoch, save):
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
        loss += get_loss_per_iter(sim, epoch, step, save=save)
    loss /= steps
    #return loss + reg.cuda()
    return loss

def do_train(cur_step,optimizer,sim):
    epoch = 0
    loss = float('inf')
    thresh = 0.00165
    num_steps_to_run = total_steps
    while True:
        
        reset_sim(sim, epoch)
        
        st = time.time()
        loss = run_sim(num_steps_to_run, sim, epoch, save=(epoch%2==0))
        
        if loss < thresh:
            print('here', loss)
            run_sim(total_steps, sim, epoch, save=True)
            break
        #if epoch > 10:
        #    loss = float('inf')
        #    break

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
    
    #initial_probs = torch.tensor([0.33,0.33,0.33],dtype=torch.float64)
    initial_probs = torch.ones((len(materials)),dtype=torch.float64)
    initial_probs /= len(initial_probs)
    print(initial_probs)
    param_g = torch.log(initial_probs)
    param_g.requires_grad = True
    lr = 0.2
    optimizer = torch.optim.Adam([param_g],lr=lr)
    for cur_step in range(tot_step):
        do_train(cur_step,optimizer,sim)
    
    #out_dir = 'exps_lasso_materialest'
    #if not os.path.exists(out_dir):
    #    os.mkdir(out_dir)
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


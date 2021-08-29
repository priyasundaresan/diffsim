import torch
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

device = torch.device("cuda:0")

handles = [0,2,38,39]
#handles = [0,2]
#handles = [0,2]

print(sys.argv)
if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

torch_model_path = out_path + ('/net_weight.pth%s'%timestamp)

class Net(nn.Module):
	def __init__(self, n_input, n_output):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(n_input, 50).double()
		self.fc2 = nn.Linear(50, 200).double()
		self.fc3 = nn.Linear(200, n_output).double()
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		# x = torch.clamp(x, min=-5, max=5)
		return x

with open('conf/rigidcloth/cloth_hang/start_jacket.json','r') as f:
	config = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')


torch.set_num_threads(8)
spf = config['frame_steps']
total_steps = 40
num_points = 5000

scalev=1

def reset_sim(sim, epoch):
    arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)

def plot_pointcloud(points, title=""):
    x, z, y = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, y, s=0.15)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_xlim([-1,1])
    ax.set_ylim([-0.5,1.0])
    ax.set_zlim([-1,0.5])
    ax.set_title(title)
    ax.view_init(30, 30)
    plt.savefig(title)
    plt.clf()
    #plt.show()

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
        ax.view_init(30, 30)
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

def get_loss_per_iter(sim, epoch, sim_step):
    curr_mesh = get_render_mesh_from_sim(sim)
    curr_pcl = sample_points_from_meshes(curr_mesh, num_points)
    ref_pcl = torch.from_numpy(np.load('demo_exp_learn_jacket_hang_pcl_video/%03d.npy'%sim_step)).to(device)
    #loss_chamfer, _ = chamfer_distance(ref_pcl, curr_pcl)
    loss_chamfer, _ = chamfer_distance(curr_pcl, ref_pcl)
    if epoch % 5 == 0:
        plot_pointclouds([curr_pcl, ref_pcl], title='%s/epoch%02d-%03d'%(out_path,epoch,sim_step))
    return loss_chamfer

def run_sim(steps, sim, net, epoch):
    loss = 0
    for param in net.parameters():
        print(torch.median(torch.abs(param.grad)).item() if param.grad is not None else None)
    updates = 0
    for step in range(steps):
        print(step)
        remain_time = torch.tensor([(total_steps - step)/total_steps],dtype=torch.float64)
        net_input = []
        for i in range(len(handles)):
        	net_input.append(sim.cloths[0].mesh.nodes[handles[i]].x)
        	net_input.append(sim.cloths[0].mesh.nodes[handles[i]].v)
        
        net_input.append(remain_time)
        net_output = net(torch.cat(net_input))
        for i in range(len(handles)):
            #sim_input = torch.cat([torch.tensor([0],dtype=torch.float64), net_output[2*i:2*i+2]])
            #sim_input = torch.cat([torch.tensor([0],dtype=torch.float64), net_output])
            sim_input = net_output[3*i:3*i+3]
            sim.cloths[0].mesh.nodes[handles[i]].v += sim_input 
        arcsim.sim_step()
        loss += get_loss_per_iter(sim, epoch, step)
        #get_loss_per_iter(sim, epoch, step)
        updates += 1
        #get_loss_per_iter(sim, epoch, step)
    #loss += get_loss_per_iter(sim, epoch, step)
    loss /= (updates)
    return loss

def do_train(cur_step,optimizer,sim,net):
    epoch = 0
    loss = float('inf')
    #thresh = 0.006 
    #thresh = 0.008 
    #thresh = 0.015 
    thresh = 0.005
    num_steps_to_run = 1
    while True:
        
        reset_sim(sim, epoch)
        
        st = time.time()
        loss = run_sim(num_steps_to_run, sim, net, epoch)
        
        if loss < thresh:
            #num_steps_to_run += 1 if num_steps_to_run < 20 else 3
            #num_steps_to_run += 5 if num_steps_to_run < 20 else 1
            num_steps_to_run += 7
        if num_steps_to_run >= total_steps:
            break
        
        en0 = time.time()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        en1 = time.time()
        print("=======================================")
        print('epoch {}: loss={}, thresh={}\n'.format(epoch, loss.data, thresh))
        
        print('forward tim = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))
        
        if epoch % 5 == 0:
        	torch.save(net.state_dict(), torch_model_path)
        
        optimizer.step()
        if epoch>=400:
        	quit()
        epoch = epoch + 1
        # break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()
	# reset_sim(sim)

	net = Net(len(handles)*6 + 1, 3*len(handles))
	if os.path.exists(torch_model_path):
		net.load_state_dict(torch.load(torch_model_path))
		print("load: %s\n success" % torch_model_path)

	#lr = 0.01
	lr = 0.01
	momentum = 0.9
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	#optimizer = torch.optim.SGD([{'params':net.parameters(),'lr':lr}],momentum=momentum)
	#optimizer = torch.optim.Adam(net.parameters(),lr=lr, weight_decay=1e-4)
	optimizer = torch.optim.Adam(net.parameters(),lr=lr)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim,net)

print("done")


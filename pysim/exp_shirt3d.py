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

handles = [41,44,48,92,103,106,113,150]
#handles = [41,44,103,106]
#handles += [163, 198]
#handles = [41,44,103,106]

if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

torch_model_path = out_path + ('/net_weight.pth%s'%timestamp)

'''
Notes: gravity -1.8 didn't error, but too little grav to learn 
made crewned3d final bigger
'''

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

with open('conf/rigidcloth/clothing/tshirt_start.json','r') as f:
	config = json.load(f)

device = torch.device("cuda:0")
# PYTORCH3D
ref_verts, ref_faces, _ = load_obj("meshes/rigidcloth/clothing/crewneck_3d_final.obj")
ref_faces_idx = ref_faces.verts_idx.to(device)
ref_verts = ref_verts.to(device)
ref_mesh = Meshes(verts=[ref_verts], faces=[ref_faces_idx])

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')

torch.set_num_threads(8)
spf = config['frame_steps']

scalev=1

def reset_sim(sim, epoch):
	if epoch % 5==0:
		arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)
	else:
		arcsim.init_physics(out_path+'/conf.json',out_path+'/out',False)

def plot_pointcloud(points, title=""):
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

def get_loss(sim):
    loss = 0
    node_number = ref_verts.shape[0]
    for i in range(node_number):
        vertex_loss = torch.norm(ref_verts[i].cpu()-(sim.cloths[0].mesh.nodes[i].x))**2
        vertex_loss *= 4 if i in handles else 1
        loss += vertex_loss
    loss /= node_number

    #verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    #faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)

    #curr_mesh = Meshes(verts=[verts], faces=[faces])

    #sample_trg = sample_points_from_meshes(ref_mesh, 9000)
    #sample_src = sample_points_from_meshes(curr_mesh, 9000)

    #loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

    #loss = loss_chamfer
    return loss

def run_sim(steps, sim, net):
    
    for step in range(steps):
        print(step)
        remain_time = torch.tensor([(steps - step)/steps],dtype=torch.float64)
        
        net_input = []
        for i in range(len(handles)):
        	net_input.append(sim.cloths[0].mesh.nodes[handles[i]].x)
        	net_input.append(sim.cloths[0].mesh.nodes[handles[i]].v)
        
        net_input.append(remain_time)
        net_output = net(torch.cat(net_input))
        
        for i in range(len(handles)):
            #sim_input = torch.cat([torch.tensor([0, 0],dtype=torch.float64), net_output[i].view([1])])
            sim_input = net_output[i:i+3]
            sim.cloths[0].mesh.nodes[handles[i]].v += sim_input 
        
        if step == 15:
            for node in sim.cloths[0].mesh.nodes:
                node.v  += torch.tensor([0, 0, 0.3],dtype=torch.float64)
        
        arcsim.sim_step()
    
    loss = get_loss(sim)
    
    return loss

def do_train(cur_step,optimizer,sim,net):
    epoch = 0
    while epoch < 81:
        steps = 25
        
        reset_sim(sim, epoch)
        
        st = time.time()
        loss = run_sim(steps, sim, net)
        en0 = time.time()
        
        optimizer.zero_grad()
        
        physics_success = True
        try:
            loss.backward()
        except:
            physics_success = False
            print("Error: failed doing loss backward. Loss:", loss)
        
        en1 = time.time()
        print("=======================================")
        print('epoch {}: loss={}\n'.format(epoch, loss.data))
        
        print('forward tim = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))
        
        if epoch % 5 == 0:
        	torch.save(net.state_dict(), torch_model_path)
        
        #if loss<1e-3:
        #	break
        
        if physics_success:
            optimizer.step()
        if epoch>=400:
        	quit()
        epoch = epoch + 1
        # break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()

	net = Net(len(handles)*6 + 1, len(handles)*3)
	#net = Net(len(handles)*6 + 1, len(handles))
	if os.path.exists(torch_model_path):
		net.load_state_dict(torch.load(torch_model_path))
		print("load: %s\n success" % torch_model_path)

	lr = 0.01
	momentum = 0.9
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	optimizer = torch.optim.Adam(net.parameters(),lr=lr)
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim,net)

print("done")

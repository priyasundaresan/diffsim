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
from scipy.spatial.transform import Rotation as R

device = torch.device("cuda:0")

#handles = [48]
#handles = [3,37,48]
#handles = [0,3,28,37,48,61,62]
handles = [0,3,28,37,48]

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
		#self.fc1 = nn.Linear(n_input, 400).double()
		#self.fc2 = nn.Linear(400, 300).double()
		#self.fc3 = nn.Linear(300, n_output).double()
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		# x = torch.clamp(x, min=-5, max=5)
		return x

with open('conf/rigidcloth/fling/start.json','r') as f:
#with open('conf/rigidcloth/fling/demo.json','r') as f:
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
    for i,points in enumerate(pcls):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
        ax.scatter3D(x, y, z, s=0.15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([0.4,1.0])
        ax.set_ylim([-0.4,0.4])
        ax.set_zlim([0,0.4])
        ax.set_title(titles[i])
        ax.view_init(30, -50)
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
    curr_mesh = get_render_mesh_from_sim(sim)
    curr_pcl = sample_points_from_meshes(curr_mesh, num_points)
    transformed_curr_pcl = (curr_pcl@rotation_pcl)*0.2 + translation_pcl
    
    ref_pcl = torch.from_numpy(np.load('demo_exp_learn_fling_real_pcl_video/%03d.npy'%(sim_step))).to(device).unsqueeze(0).float()
    loss_chamfer, _ = chamfer_distance(transformed_curr_pcl, ref_pcl)
    if save:
        plot_pointclouds([transformed_curr_pcl, ref_pcl], title='%s/epoch%02d-%03d'%(out_path,epoch,sim_step))
    return loss_chamfer

def run_sim(steps, sim, net, epoch, save):
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
            sim_input = net_output[3*i:3*i+3]
            sim.cloths[0].mesh.nodes[handles[i]].v += sim_input 
        arcsim.sim_step()
        loss += get_loss_per_iter(sim, epoch, step, save=save)
        updates += 1
    loss /= (updates)
    return loss

def do_train(cur_step,optimizer,sim,net):
    epoch = 0
    loss = float('inf')
    thresh = 0.009
    num_steps_to_run = 1
    while True:
        
        reset_sim(sim, epoch)
        
        st = time.time()
        loss = run_sim(num_steps_to_run, sim, net, epoch, save=(epoch%10==0))
        
        if num_steps_to_run >= total_steps:
            run_sim(num_steps_to_run, sim, net, epoch, save=True)
            break
        if loss < thresh:
            num_steps_to_run += 1
    
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

	net = Net(len(handles)*6 + 1, len(handles*3))
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


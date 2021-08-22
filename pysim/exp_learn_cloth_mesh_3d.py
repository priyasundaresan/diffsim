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

device = torch.device("cuda:0")
verts, faces, aux = load_obj("conf/rigidcloth/drag/final_target_cloth.obj")
faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)
ref_target_cloth_mesh = Meshes(verts=[verts], faces=[faces_idx])

handles = [25, 60, 30, 54]

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

with open('conf/rigidcloth/drag/drag.json','r') as f:
	config = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')


torch.set_num_threads(8)
spf = config['frame_steps']

scalev=1

def reset_sim(sim, epoch):
    arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)

def get_loss(sim):
    loss = 0
    node_number = verts.shape[0]
    for i in range(node_number):
        vertex_loss = torch.norm(verts.cpu()[i]-(sim.cloths[0].mesh.nodes[i].x))**2
        loss += vertex_loss
    loss /= node_number

    return loss

def run_sim(steps, sim, net):
    #for param in net.parameters():
    #    print(param.grad)
    for obstacle in sim.obstacles:
    	for node in obstacle.curr_state_mesh.nodes:
    		node.m    *= 0.2
    
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
            sim_input = net_output[3*i:3*i + 3]
            #print(sim_input.grad)
            sim.cloths[0].mesh.nodes[handles[i]].v += sim_input 
        
        arcsim.sim_step()
    
    loss = get_loss(sim)
    
    return loss

def do_train(cur_step,optimizer,sim,net):
	epoch = 0
	while True:
		# steps = int(1*15*spf)
		steps = 20

		reset_sim(sim, epoch)

		st = time.time()
		loss = run_sim(steps, sim, net)
		en0 = time.time()
		
		optimizer.zero_grad()

		loss.backward()

		en1 = time.time()
		print("=======================================")
		print('epoch {}: loss={}\n'.format(epoch, loss.data))

		print('forward tim = {}'.format(en0-st))
		print('backward time = {}'.format(en1-en0))


		if epoch % 5 == 0:
			torch.save(net.state_dict(), torch_model_path)

		if loss<1e-3:
			break
		# dgrad, stgrad, begrad = torch.autograd.grad(loss, [density, stretch, bend])

		optimizer.step()
		if epoch>=400:
			quit()
		epoch = epoch + 1
		# break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()
	# reset_sim(sim)

	#param_g = torch.tensor([0,0,0,0,0,1],dtype=torch.float64, requires_grad=True)
	net = Net(25, len(handles)*3)
	if os.path.exists(torch_model_path):
		net.load_state_dict(torch.load(torch_model_path))
		print("load: %s\n success" % torch_model_path)

	lr = 0.01
	momentum = 0.9
	f.write('lr={} momentum={}\n'.format(lr,momentum))
	#optimizer = torch.optim.SGD([{'params':net.parameters(),'lr':lr}],momentum=momentum)
	optimizer = torch.optim.Adam(net.parameters(),lr=lr)
	# optimizer = torch.optim.Adadelta([density, stretch, bend])
	for cur_step in range(tot_step):
		do_train(cur_step,optimizer,sim,net)

print("done")


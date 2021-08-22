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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

import cv2
import matplotlib.image as mpimg

from pytorch3d.renderer import (
    look_at_view_transform,
    look_at_rotation,
    FoVPerspectiveCameras, 
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

device = torch.device("cuda:0")
lights = DirectionalLights(device=device, direction=((0,-1.0,0),))
R, T = look_at_view_transform(1, 300, 0) 
T[0][0] += 0.5
T[0][1] -= 0.1

camera = FoVPerspectiveCameras(device=device, R=R, T=T)

raster_settings = RasterizationSettings(
    image_size=300, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
    perspective_correct=False
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=camera,
        lights=lights
    )
)

criterion = torch.nn.MSELoss(reduction='mean')

handles = [25, 60, 30, 54]

total_steps = 20

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

with open('conf/rigidcloth/drag/drag_plane_below.json','r') as f:
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

def get_render_mesh_from_sim(sim):
    colors = torch.Tensor([[1,0,0], [0,1,0], [0,0,1]])

    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)
    cloth_rgb = torch.ones_like(cloth_verts) # (V, 3)
    cloth_rgb[:,] = colors[2]

    all_verts = [cloth_verts]
    all_faces = [cloth_faces]
    all_textures = [cloth_rgb]
    all_tex = torch.cat(all_textures)[None]
    textures = TexturesVertex(verts_features=all_tex)

    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)], textures=textures)
    return mesh

def get_loss_per_iter(sim, epoch, sim_iter):
    curr_mesh = get_render_mesh_from_sim(sim)
    curr_image = renderer(curr_mesh)[0,...,:3]
    ref_image = mpimg.imread('demo_exp_learn_cloth_image/%03d.jpg'%sim_iter)
    ref_image = torch.from_numpy(ref_image)[...,:3].to(device)/255.

    if epoch % 1 == 0:
        visualization = np.hstack((curr_image.detach().cpu().numpy(), ref_image.detach().cpu().numpy()))
        cv2.imwrite('%s/epoch%03d-%03d.jpg'%(out_path, epoch, sim_iter), visualization*255)

    loss = criterion(curr_image, ref_image) 
    return loss

def run_sim(steps, sim, net, epoch):

    #for param in net.parameters():
    #    print(param.grad)

    for obstacle in sim.obstacles:
    	for node in obstacle.curr_state_mesh.nodes:
    		node.m    *= 0.2
    
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
            sim_input = torch.cat([torch.tensor([0, 0],dtype=torch.float64), net_output[i].view([1])])
            #print(sim_input.grad)
            sim.cloths[0].mesh.nodes[handles[i]].v += sim_input 
        
        arcsim.sim_step()
    
    loss = get_loss_per_iter(sim, epoch, step)
    return loss

def do_train(cur_step,optimizer,sim,net):
	epoch = 0
	while True:

		reset_sim(sim, epoch)

		st = time.time()
		loss = run_sim(total_steps, sim, net, epoch)
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
	net = Net(25, 4)
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


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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

import cv2

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

handles = [60,30]

out_path = 'default_out'
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

with open('conf/rigidcloth/triangle_fold/start.json','r') as f:
	config = json.load(f)

# PYTORCH3D
device = torch.device("cuda:0")

verts, faces, aux = load_obj("meshes/rigidcloth/fold_target/triangle_fold.obj")
faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)
verts_rgb = torch.ones_like(verts)[None] # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))
ref_mesh = Meshes(verts=[verts], faces=[faces_idx], textures=textures)

num_views = 1

# Get a batch of viewing angles. 
lights = DirectionalLights(device=device, direction=((0,-1.0,0),))
R0, T0 = look_at_view_transform(2.7, 180, 0) # top down
R1, T1 = look_at_view_transform(2.7, 270, 0) 
R2, T2 = look_at_view_transform(2.7, 90, 0)
camera0 = FoVPerspectiveCameras(device=device, R=R0, T=T0)
camera1 = FoVPerspectiveCameras(device=device, R=R1, T=T1)
camera2 = FoVPerspectiveCameras(device=device, R=R2, T=T2)

# We arbitrarily choose one particular view that will be used to visualize 
# results

raster_settings = RasterizationSettings(
    image_size=300, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
    perspective_correct=False
)

renderer0 = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera0, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=camera0,
        lights=lights
    )
)

renderer1 = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera1, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=camera1,
        lights=lights
    )
)

renderer2 = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera2, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=camera2,
        lights=lights
    )
)

criterion = torch.nn.MSELoss(reduction='mean')

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

def get_loss(sim, epoch):

    verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)

    curr_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    curr_image_0 = renderer0(curr_mesh)[0,...,:3]
    ref_image_0 = renderer0(ref_mesh)[0,...,:3]

    curr_image_1 = renderer1(curr_mesh)[0,...,:3]
    ref_image_1 = renderer1(ref_mesh)[0,...,:3]

    curr_image_2 = renderer2(curr_mesh)[0,...,:3]
    ref_image_2 = renderer2(ref_mesh)[0,...,:3]
    
    if epoch % 2 == 0:
        visualization0 = np.hstack((curr_image_0.detach().cpu().numpy(), ref_image_0.detach().cpu().numpy()))
        visualization1 = np.hstack((curr_image_1.detach().cpu().numpy(), ref_image_1.detach().cpu().numpy()))
        visualization2 = np.hstack((curr_image_2.detach().cpu().numpy(), ref_image_2.detach().cpu().numpy()))
        visualization = np.vstack((visualization0, visualization1, visualization2))
        #cv2.imshow('img', renderer1(curr_mesh)[0,...,3].detach().cpu().numpy())
        #cv2.waitKey(0)
        #cv2.imshow('img', renderer2(curr_mesh)[0,...,3].detach().cpu().numpy())
        #cv2.waitKey(0)
        #cv2.imshow('img', visualization)
        #cv2.waitKey(0)
        cv2.imwrite('%s/epoch%05d.jpg'%(out_path, epoch), visualization*255)

    #cv2.imshow("img", visualization)
    #cv2.waitKey(0)

    sample_trg = sample_points_from_meshes(ref_mesh, 1000)
    sample_src = sample_points_from_meshes(curr_mesh, 1000)

    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

    #loss = loss_chamfer
    loss0 = criterion(curr_image_0, ref_image_0)
    loss1 = criterion(curr_image_1, ref_image_1)
    loss2 = criterion(curr_image_2, ref_image_2)
    loss = (loss0+loss1+loss2)/3
    print("chamfer loss:", loss_chamfer)
    return loss

def run_sim(steps, sim, net, epoch):

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
			sim_input = torch.cat([torch.tensor([0, 0],dtype=torch.float64), net_output[i].view([1])])
			sim.cloths[0].mesh.nodes[handles[i]].v += sim_input 

		arcsim.sim_step()

	loss = get_loss(sim, epoch)

	return loss

def do_train(cur_step,optimizer,sim,net):
    epoch = 0
    while epoch < 81:
    #while epoch < 31:
    	# steps = int(1*15*spf)
    	steps = 25
    
    	reset_sim(sim, epoch)
    
    	st = time.time()
    	#loss, ans = run_sim(steps, sim, net, goal)
    	loss = run_sim(steps, sim, net, epoch)
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
    
    	optimizer.step()
    	if epoch>=400:
    		quit()
    	epoch = epoch + 1
    	# break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
	tot_step = 1
	sim=arcsim.get_sim()

	net = Net(len(handles)*6 + 1, len(handles))
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

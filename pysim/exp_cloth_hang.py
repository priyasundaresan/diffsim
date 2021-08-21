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
import matplotlib.image as mpimg

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

handles = [44]
epochs = 40 
steps = 100

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
        return x

with open('conf/rigidcloth/cloth_hang/start.json','r') as f:
	config = json.load(f)

# PYTORCH3D
device = torch.device("cuda:0")

num_views = 1

# Get a batch of viewing angles. 
lights = DirectionalLights(device=device, direction=((0,-1.0,0),))
R, T = look_at_view_transform(1.25, 300, 0) 

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

def get_render_mesh_from_sim(sim):
    colors = torch.Tensor([[1,0,0], [0,1,0], [0,0,1]])

    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)
    cloth_rgb = torch.ones_like(cloth_verts) # (V, 3)
    cloth_rgb[:,] = colors[2]

    ground_verts = torch.stack([v.node.x for v in sim.obstacles[0].curr_state_mesh.verts]).float().to(device)
    ground_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.obstacles[0].curr_state_mesh.faces]).to(device)
    ground_faces += len(cloth_verts)
    ground_rgb = torch.ones_like(ground_verts) # (V, 3)
    ground_rgb[:,] = colors[1]

    pole_verts = torch.stack([v.node.x for v in sim.obstacles[1].curr_state_mesh.verts]).float().to(device)
    pole_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.obstacles[1].curr_state_mesh.faces]).to(device)
    pole_faces += len(cloth_verts) + len(ground_verts)
    pole_rgb = torch.ones_like(pole_verts) # (V, 3)
    pole_rgb[:,] = colors[0]

    colors = torch.Tensor([[1,0,0], [0,1,0], [0,0,1]])
    all_verts = [cloth_verts, ground_verts, pole_verts]
    all_faces = [cloth_faces, ground_faces, pole_faces]
    all_textures = [cloth_rgb, ground_rgb, pole_rgb]
    all_tex = torch.cat(all_textures)[None]
    textures = TexturesVertex(verts_features=all_tex)

    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)], textures=textures)
    
    return mesh

def get_loss_per_iter(sim, epoch, sim_iter):

    curr_mesh = get_render_mesh_from_sim(sim)

    curr_image = renderer(curr_mesh)[0,...,:3]
    ref_image = mpimg.imread('demo_video_frames/%03d.jpg'%sim_iter)
    ref_image = torch.from_numpy(ref_image)[...,:3].to(device)/255.

    if epoch % 2 == 0:
        visualization = np.hstack((curr_image.detach().cpu().numpy(), ref_image.detach().cpu().numpy()))
        cv2.imwrite('%s/epoch%03d-%03d.jpg'%(out_path, epoch, sim_iter), visualization*255)

    loss = criterion(curr_image, ref_image) 
    return loss

def run_sim(steps, sim, net, epoch):
    loss = 0
    for step in range(steps):
        remain_time = torch.tensor([(steps - step)/steps],dtype=torch.float64)
        net_input = []
        for i in range(len(handles)):
        	net_input.append(sim.cloths[0].mesh.nodes[handles[i]].x)
        	net_input.append(sim.cloths[0].mesh.nodes[handles[i]].v)
        net_input.append(remain_time)
        net_output = net(torch.cat(net_input))
        
        for i in range(len(handles)):
            sim_input = net_output[i:i+3]
            print(sim_input)
            sim.cloths[0].mesh.nodes[handles[i]].v += sim_input 
        arcsim.sim_step()
        loss += get_loss_per_iter(sim, epoch, step)
    loss /= steps
    return loss

def do_train(cur_step,optimizer,sim,net):
    epoch = 0
    while epoch < epochs:
        reset_sim(sim, epoch)
        st = time.time()
        lst = list(range(2, steps, 2))
        prog = (epoch/epochs) * steps
        num_steps_to_run = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-prog))]
        #num_steps_to_run = steps
        #loss = run_sim(steps, sim, net, epoch)
        loss = run_sim(num_steps_to_run, sim, net, epoch)
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
        
        optimizer.step()
        if epoch>=400:
        	quit()
        epoch = epoch + 1
        # break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
    tot_step = 1
    sim=arcsim.get_sim()
    
    net = Net(len(handles)*6 + 1, len(handles)*3)
    if os.path.exists(torch_model_path):
    	net.load_state_dict(torch.load(torch_model_path))
    	print("load: %s\n success" % torch_model_path)
    
    lr = 0.2
    momentum = 0.9
    f.write('lr={} momentum={}\n'.format(lr,momentum))
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    for cur_step in range(tot_step):
        do_train(cur_step,optimizer,sim,net)

print("done")

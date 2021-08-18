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

from datetime import datetime
now = datetime.now()
timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

steps = 25
epochs= 40
handles = [60,30]

losses = []
param_g = torch.zeros([steps, len(handles)*3],dtype=torch.float64, requires_grad=True)

out_path = 'default_out'
if not os.path.exists(out_path):
    os.mkdir(out_path)

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
R, T = look_at_view_transform(0.85, 280, 0) 
T[0] += 0.3
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
scalev=1

def reset_sim(sim, epoch):
	if epoch % 5==0:
		arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)
	else:
		arcsim.init_physics(out_path+'/conf.json',out_path+'/out',False)

def get_loss_per_iter(sim, epoch, sim_iter):
    verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)

    curr_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    curr_image = renderer(curr_mesh)[0,...,:3]
    ref_image = mpimg.imread('demo_video_frames/%03d.jpg'%sim_iter)
    ref_image = torch.from_numpy(ref_image)[...,:3].to(device)/255.

    if epoch % 2 == 0 and sim_iter==24:
        visualization = np.hstack((curr_image.detach().cpu().numpy(), ref_image.detach().cpu().numpy()))
        cv2.imwrite('%s/epoch%05d.jpg'%(out_path, epoch), visualization*255)

    sample_trg = sample_points_from_meshes(ref_mesh, 1000)
    sample_src = sample_points_from_meshes(curr_mesh, 1000)

    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

    loss = criterion(curr_image, ref_image) 
    if sim_iter == 24:
        print("chamfer loss", loss_chamfer)
    return loss

#def get_loss(sim,ref):
#    reg  = torch.norm(param_g, p=2)*0.001
#    loss = 0
#    print("VERTS", ref.shape[0], len(sim.cloths[0].mesh.nodes))
#
#    for i in range(ref.shape[0]):
#        loss += torch.norm(ref[i]-sim.cloths[0].mesh.nodes[i].x)**2
#    loss /= node_number
#
#    loss += reg
#    return loss

def run_sim(steps, sim, epoch):
    loss = 0
    for step in range(steps):
        for i in range(len(handles)):
            inc_v = param_g[step,3*i:3*i+3]
            sim.cloths[0].mesh.nodes[handles[i]].v += inc_v
            del inc_v
        arcsim.sim_step()
        loss += get_loss_per_iter(sim, epoch, step)
    return loss

def do_train(cur_step,optimizer,scheduler,sim):
    epoch = 0
    while epoch < epochs:
        reset_sim(sim, epoch)
        st = time.time()
        loss = run_sim(steps, sim, epoch)
        en0 = time.time()
        optimizer.zero_grad()
        loss.backward()
        en1 = time.time()
        print("=======================================")
        f.write('epoch {}:  loss={} \n'.format(epoch,  loss.data))
        print('epoch {}:  loss={} \n'.format(epoch, loss.data))
        print('forward time={}'.format(en0-st))
        print('backward time={}'.format(en1-en0))
        optimizer.step()
        losses.append(loss)
        epoch = epoch + 1

def visualize_loss(losses,dir_name):
    plt.plot(losses)
    plt.title('losses')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.savefig(dir_name+'/'+'loss.jpg')

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
    tot_step = 1
    sim=arcsim.get_sim()
    lr = 80
    momentum = 0.9
    f.write('lr={} momentum={}\n'.format(lr,momentum))
    optimizer = torch.optim.SGD([{'params':param_g,'lr':lr}],momentum=momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10,2,eta_min=0.0001)
    for cur_step in range(tot_step):
        do_train(cur_step,optimizer,scheduler,sim)
    visualize_loss(losses,out_path)

print("done")

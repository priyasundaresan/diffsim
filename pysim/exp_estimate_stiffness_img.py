import os
import cv2
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
from pytorch3d.io import load_obj
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


with open('conf/rigidcloth/sysid/start.json','r') as f:
	config = json.load(f)

# PYTORCH3D
device = torch.device("cuda:0")
ref_mesh = load_objs_as_meshes(["meshes/rigidcloth/sysid/ref_pinned_less_stiff.obj"], device=device)
#ref_mesh = load_objs_as_meshes(["meshes/rigidcloth/sysid/ref_pinned_more_stiff.obj"], device=device)
white_tex = torch.ones_like(ref_mesh.verts_packed()).unsqueeze(0) 
white_tex = TexturesVertex(verts_features=white_tex.to(device))
ref_mesh.textures = white_tex

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')

torch.set_num_threads(8)
spf = config['frame_steps']

scalev=1

sigma = 1e-4
R, T = look_at_view_transform(1.5, -85, 0) 
T[0][0] += 0.5
camera = OpenGLPerspectiveCameras(device=device, R=R, T=T)
lights = DirectionalLights(device=device, direction=((1,0,0),))
raster_settings = RasterizationSettings(
    image_size=128, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
    perspective_correct=False
)
raster_settings_soft = RasterizationSettings(
    image_size=128, 
    blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
    faces_per_pixel=50, 
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
renderer_soft = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings_soft
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=camera,
        lights=lights
    )
)
renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader()
)
renderer_silhouette_soft = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=camera, 
        raster_settings=raster_settings_soft
    ),
    shader=SoftSilhouetteShader()
)

def reset_sim(sim, epoch):
	arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)

def get_render_mesh_from_sim(sim):
    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)
    cloth_rgb = torch.ones_like(cloth_verts) # (V, 3)

    all_verts = [cloth_verts]
    all_faces = [cloth_faces]
    all_textures = [cloth_rgb]
    tex = torch.cat(all_textures)[None]
    textures = TexturesVertex(verts_features=tex.to(device))

    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)], textures=textures)
    return mesh

def get_loss(sim, epoch):
    curr_mesh = get_render_mesh_from_sim(sim)
    reg  = torch.norm(param_g, p=2)*0.001
    loss = 0
    curr_image = renderer_soft(curr_mesh, cameras=camera, lights=lights)[..., :3]
    curr_sil = renderer_silhouette_soft(curr_mesh, cameras=camera, lights=lights)[..., 3]
    ref_image = renderer(ref_mesh, cameras=camera, lights=lights)[..., :3]
    ref_sil = renderer_silhouette(ref_mesh, cameras=camera, lights=lights)[..., 3]
    loss_rgb = (torch.abs(curr_image - ref_image)).mean()
    loss_silhouette = (torch.abs(curr_sil - ref_sil)).mean()
    #loss = loss_rgb*0.5 + loss_silhouette*0.5
    loss = loss_rgb*0.0 + loss_silhouette*1.0
    visualization = np.hstack((curr_image[0].detach().cpu().numpy(), ref_image[0].detach().cpu().numpy()))
    cv2.imwrite('%s/epoch%03d.jpg'%(out_path, epoch), visualization*255)
    return loss.cpu() + reg

def run_sim(steps, sim, epoch):
    output_processed = param_g
    print("param_g:", output_processed)
    orig = sim.cloths[0].materials[0].stretching
    sim.cloths[0].materials[0].stretching = orig*output_processed.cpu()
    for step in range(steps):
        print(step)
        arcsim.sim_step()
    loss = get_loss(sim, epoch)
    return loss

def do_train(cur_step,optimizer,sim):
    epoch = 0
    while True:
        steps = 40
        
        reset_sim(sim, epoch)
        
        st = time.time()
        loss = run_sim(steps, sim, epoch)
        en0 = time.time()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        if loss < 0.016:
        #if loss < 0.02:
            break
        
        en1 = time.time()
        print("=======================================")
        print('epoch {}: loss={}\n'.format(epoch, loss.data))
        
        print('forward tim = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))
        
        
        optimizer.step()
        epoch = epoch + 1
# break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
    tot_step = 1
    sim=arcsim.get_sim()
    
    param_g = torch.tensor(0.5,dtype=torch.float64, requires_grad=True)
    
    lr = 0.1
    #lr = 0.15
    optimizer = torch.optim.Adam([param_g],lr=lr)
    for cur_step in range(tot_step):
    	do_train(cur_step,optimizer,sim)

print("done")

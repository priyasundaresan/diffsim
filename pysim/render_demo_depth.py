import torch
import matplotlib.pyplot as plt
import numpy as np
import os
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
from pytorch3d.io import load_obj, load_objs_as_meshes

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
    SoftGouraudShader,
    TexturesUV,
    TexturesVertex
)

device = torch.device("cuda:0")

lights = DirectionalLights(device=device, direction=((1,0,1),))
#R, T = look_at_view_transform(1.5, -80, 0) 
R, T = look_at_view_transform(1.5, 0, 0) 

camera = FoVPerspectiveCameras(device=device, R=R, T=T)
raster_settings = RasterizationSettings(
    image_size=150, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
    perspective_correct=True
)

rasterizer=MeshRasterizer(
    cameras=camera, 
    raster_settings=raster_settings
)

num_frames = 30
demo_length = 30
step = demo_length//num_frames
out_dir = 'demo_video_frames'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for i in range(0, demo_length, step):
    mesh_fnames = sorted([f for f in os.listdir('default_out/out0') if '%04d'%i in f])
    all_verts = []
    all_faces = []
    all_textures = []
    vert_count = 0
    for j, f in enumerate(mesh_fnames[:1]):
    #for j, f in enumerate(mesh_fnames):
        verts, faces, aux = load_obj(os.path.join("default_out", "out0", f))
        faces_idx = faces.verts_idx.to(device) + vert_count
        verts = verts.to(device)
        vert_count += len(verts)
        verts_rgb = torch.ones_like(verts) # (V, 3)
        all_verts.append(verts)
        all_faces.append(faces_idx)
        all_textures.append(verts_rgb)
    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)])
    fragments = rasterizer(mesh)
    img = fragments.zbuf[0].squeeze()
    minval, maxval = 1.3177, 1.5088
    #minval, maxval = torch.min(img), torch.max(img)
    #print(torch.min(img[torch.where(img != -1)]))
    #minval, maxval = 0.8555, 1.9296
    #img[torch.where(img==-1)] = minval
    #print(minval, maxval)
    #img[torch.where(img==-1)] = minval
    img = (img - minval)/(maxval - minval)
    #img = torch.ones_like(img) - img
    img = img.cpu().numpy()
    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    cv2.imwrite('%s/%03d.jpg'%(out_dir, i//step), img*255)

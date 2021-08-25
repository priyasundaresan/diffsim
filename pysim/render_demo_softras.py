import torch
import os
import cv2
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse

import soft_renderer as sr


device = torch.device("cuda:0")
camera_distance = 1.25
elevation = 20
azimuth = -20

transform = sr.LookAt()
lighting = sr.Lighting()
#rasterizer = sr.SoftRasterizer()
rasterizer = sr.SoftRasterizer(image_size=64, sigma_val=1e-4, aggr_func_rgb='hard')

num_frames = 30
demo_length = 30
step = demo_length//num_frames
out_dir = 'demo_video_frames'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for i in range(0, demo_length, step):
    mesh_fnames = sorted([f for f in os.listdir('default_out/out0') if '%04d'%i in f])
    print(mesh_fnames)
    all_verts = []
    all_faces = []
    all_textures = []
    vert_count = 0
    colors = torch.Tensor([[1,0,0], [0,1,0], [0,0,1]])
    for j, f in enumerate(mesh_fnames):
        mesh_ = sr.Mesh.from_obj(os.path.join("default_out", "out0", f))
        verts, faces_idx = mesh_.vertices, mesh_.faces
        faces_idx = faces_idx.to(device).squeeze() + vert_count
        verts = verts.to(device).squeeze()
        vert_count += len(verts)
        verts_rgb = torch.ones_like(verts)
        verts_rgb[:,] = colors[j]
        verts = torch.stack((verts[:,0], verts[:,2], verts[:,1])).T
        all_verts.append(verts)
        all_faces.append(faces_idx)
        all_textures.append(verts_rgb)
    tex = torch.cat(all_textures)
    vertices = torch.cat(all_verts).unsqueeze(0)
    faces = torch.cat(all_faces).unsqueeze(0)
    mesh = sr.Mesh(vertices=vertices, faces=faces, textures=tex, texture_type='vertex')
    mesh = lighting(mesh)
    transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
    mesh = transform(mesh)
    images = rasterizer(mesh)
    image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
    cv2.imwrite('%s/%03d.jpg'%(out_dir, i//step), image*255)
    #cv2.imshow('img', image)
    #cv2.waitKey(0)
    #img  = renderer(mesh)[0,...,:3]*255
    #cv2.imwrite('%s/%03d.jpg'%(out_dir, i//step), img.detach().cpu().numpy())


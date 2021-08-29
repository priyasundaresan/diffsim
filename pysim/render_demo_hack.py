import torch
import os
import cv2

import pytorch3d
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

device = torch.device("cuda:0")
#device = "cpu"

from pytorch3d.renderer import (
    look_at_view_transform,
    look_at_rotation,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    OpenGLPerspectiveCameras, 
    BlendParams,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    SoftGouraudShader,
    TexturesUV,
    TexturesVertex,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

device = torch.device("cuda:0")

lights = DirectionalLights(device=device, direction=((0,-1.0,0),))
#lights = None

R, T = look_at_view_transform(1.25, -60, 0) 
camera = FoVPerspectiveCameras(device=device, R=R, T=T)
raster_settings = PointsRasterizationSettings(
    image_size=128, 
    radius = 0.005,
    points_per_pixel = 150
)
rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
renderer = PointsRenderer(
    rasterizer=rasterizer,
    compositor=AlphaCompositor()
)
print(dir(renderer))

num_frames = 30
demo_length = 30
num_points=1000000

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
    colors = torch.Tensor([[0,0,1], [0,1,0], [1,0,0]])
    for j, f in enumerate(mesh_fnames[:1]):
    #for j, f in enumerate(mesh_fnames):
        verts, faces, aux = load_obj(os.path.join("default_out", "out0", f))
        faces_idx = faces.verts_idx.to(device) + vert_count
        verts = verts.to(device)
        vert_count += len(verts)
        verts_rgb = torch.ones_like(verts) # (V, 3)
        verts_rgb[:,] = colors[j]
        all_verts.append(verts)
        all_faces.append(faces_idx)
        all_textures.append(verts_rgb)
    tex = torch.cat(all_textures)[None]
    textures = TexturesVertex(verts_features=tex.to(device))
    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)], textures=textures)
    curr_pcl, rgb = sample_points_from_meshes(mesh, num_points, return_textures=True)
    point_cloud = Pointclouds(points=[curr_pcl.squeeze()], features=[rgb.squeeze()])
    img = renderer(point_cloud)[0, ..., :3]*255
    cv2.imwrite('%s/%03d.jpg'%(out_dir, i//step), img.detach().cpu().numpy())

import torch
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
    TexturesUV,
    TexturesVertex
)

device = torch.device("cuda:0")

lights = DirectionalLights(device=device, direction=((0,-1.0,0),))
# this is for cloth folding
#R, T = look_at_view_transform(1.25, 340, 0) 
#T[0][0] += 0.4
#T[0][1] += -0.2

# this is for bag
R, T = look_at_view_transform(1.25, 300, 0) 
T[0][0] += 0.4
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
    colors = torch.Tensor([[1,0,0], [0,1,0], [0,0,1]])
    for j, f in enumerate(mesh_fnames[:1]):
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
    img  = renderer(mesh)[0,...,:3]*255
    cv2.imwrite('%s/%03d.jpg'%(out_dir, i//step), img.detach().cpu().numpy())

import torch
#import pytorch3d
import pprint
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

from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

#from pytorch3d.structures import Meshes
#from pytorch3d.ops import sample_points_from_meshes
#from pytorch3d.loss import (
#    chamfer_distance, 
#    mesh_edge_loss, 
#    mesh_laplacian_smoothing, 
#    mesh_normal_consistency,
#)
#from pytorch3d.io import load_obj

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

handles = [60,30]

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

with open('conf/rigidcloth/triangle_fold/start.json','r') as f:
	config = json.load(f)

#ref = np.load('conf/rigidcloth/triangle_fold/ref_mesh.npy')
#ref = torch.from_numpy(np.vstack(ref))

ref_verts = np.load('conf/rigidcloth/triangle_fold/ref_verts.npy')
ref_verts = torch.from_numpy(ref_verts)
ref_faces = np.load('conf/rigidcloth/triangle_fold/ref_faces.npy')
ref_faces = torch.from_numpy(ref_faces)
ref_face_areas = np.load('conf/rigidcloth/triangle_fold/ref_face_areas.npy')
ref_face_areas = torch.from_numpy(ref_face_areas)

#print(ref_verts.shape, ref_faces.shape, ref_face_areas.shape)

#device = torch.device("cuda:0")
#verts, faces, aux = load_obj("meshes/rigidcloth/fold_target/triangle_fold.obj")
#faces_idx = faces.verts_idx.to(device)
#verts = verts.to(device)
#ref_mesh = Meshes(verts=[verts], faces=[faces_idx])

device = torch.device("cuda:0")

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

def _rand_barycentric_coords(size1, size2, dtype: torch.dtype, device: torch.device):
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    return w0, w1, w2

def sample_points_from_meshes(verts, faces, areas, num_samples):
    num_meshes = 1

    # Only compute samples for non empty meshes
    with torch.no_grad():
        sample_face_idxs = areas.squeeze().multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces.long()]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    ## Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_meshes, num_samples, verts.dtype, verts.device
    )

    ## Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    #print(a.shape)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]

    points = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c
    return points

def plot_pointcloud(points, title=""):
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

def get_loss(sim):
    #reg  = torch.norm(param_g, p=2)*0.001
    loss = 0
    #node_number = ref.shape[0]

    #for v in sim.cloths[0].mesh.verts:
    #    print(v.index, v.node.x)
    #for f in sim.cloths[0].mesh.faces:
    #    print(f.index, [vert.index for vert in f.v], f.a)

    verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts])
    faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces])
    face_areas = torch.Tensor([[f.a] for f in sim.cloths[0].mesh.faces])

    ref_points = sample_points_from_meshes(ref_verts, ref_faces, ref_face_areas, 1000)
    curr_points = sample_points_from_meshes(verts, faces, face_areas, 1000)
    #plot_pointcloud(ref_points)

    #for i in range(node_number):
    #    loss += torch.norm(ref[i]-(sim.cloths[0].mesh.nodes[i].x))**2
    #loss /= node_number

    dist1, dist2 = chamfer_dist(curr_points.float(), ref_points.float())
    loss = (torch.mean(dist1)) + (torch.mean(dist2))

    #loss += reg
    return loss


def run_sim(steps, sim, net):

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

	loss = get_loss(sim)

	return loss

def do_train(cur_step,optimizer,sim,net):
	epoch = 0
	while True:
		# steps = int(1*15*spf)
		steps = 20

		reset_sim(sim, epoch)

		st = time.time()
		#loss, ans = run_sim(steps, sim, net, goal)
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


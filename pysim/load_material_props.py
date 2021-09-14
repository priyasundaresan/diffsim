import torch
import pprint
import arcsim
import gc
import time
import json
import sys
import gc
import os
import numpy as np

device = torch.device("cuda:0")

def load_material(path_to_json, device):
    with open(path_to_json,'r') as f:
    	config = json.load(f)
    density = torch.Tensor([config["density"]])
    bending = torch.Tensor(config["bending"])
    stretching = torch.Tensor(config["stretching"])
    return density, bending, stretching

def combine_materials(density, bending, stretching, proportions):
    final_density = density.dot(proportions).double()
    final_bending = torch.einsum('bij,b->bij', bending, proportions).sum(0).double()
    final_stretching = torch.einsum('bij,b->bij', stretching, proportions).sum(0).double()
    return final_density, final_bending, final_stretching

def test_lasso_sim(density, bending, stretching):
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/lasso/demo_slow.json'),'default_out/out0',False)
    sim.cloths[0].materials[0].densityori= density
    sim.cloths[0].materials[0].stretchingori = stretching
    sim.cloths[0].materials[0].bendingori = bending
    arcsim.reload_material_data(sim.cloths[0].materials[0])
    for step in range(50):
        print(step)
        arcsim.sim_step()

def test_twoin_fold_demo(density, bending, stretching):
    print('here')
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/fold/demo_fast.json'),'default_out/out0',False)
    sim.cloths[0].materials[0].densityori= density
    sim.cloths[0].materials[0].stretchingori = stretching
    sim.cloths[0].materials[0].bendingori = bending
    arcsim.reload_material_data(sim.cloths[0].materials[0])
    for step in range(5):
        arcsim.sim_step()
    for step in range(25):
        sim.cloths[0].mesh.nodes[0].v += torch.Tensor([step/2,step/2,step/10]).double()
        sim.cloths[0].mesh.nodes[3].v += torch.Tensor([-step/2,-step/2,step/10]).double()
        arcsim.sim_step()
    for step in range(5):
        arcsim.sim_step()

if __name__ == '__main__':
    #materials = ['11oz-black-denim.json', 'gray-interlock.json', 'ivory-rib-knit.json', \
    #             'navy-sparkle-sweat.json', 'white-dots-on-blk.json']
    materials = ['11oz-black-denim.json', 'gray-interlock.json', 'navy-sparkle-sweat.json']
    base_dir = 'materials'
    density_all = []
    bending_all = []
    stretching_all = []
    for m in materials:
        d,b,s = load_material(os.path.join(base_dir, m), torch.device("cuda:0")) 
        density_all.append(d)
        bending_all.append(b.tolist())
        stretching_all.append(s.tolist())
    density_all = torch.Tensor(density_all)
    bending_all = torch.Tensor(bending_all)
    stretching_all = torch.Tensor(stretching_all)
    #proportions = torch.Tensor([0.1, 0.7, 0.2])
    proportions = torch.Tensor([0.0, 0.0, 1.0])
    density, bend, stretch = combine_materials(density_all, bending_all, stretching_all, proportions)
    #test_lasso_sim(density, bend, stretch)
    test_twoin_fold_demo(density, bend, stretch)
    
    
    

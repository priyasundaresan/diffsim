import torch
import arcsim
import gc
import time
import json
import sys
import gc
import os
import numpy as np

def get_target_mesh(conf_dir, path):
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join(conf_dir, path),'default_out/target',False)
    node_number = len(sim.cloths[0].mesh.nodes)
    ref = [sim.cloths[0].mesh.nodes[i].x.numpy() for i in range(node_number)]
    np.save(os.path.join(conf_dir, 'ref_mesh.npy'), ref)
    return ref

def get_target_triangle_fold(conf_dir, path):
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join(conf_dir, path),'default_out/target',False)
    verts = np.stack([v.node.x for v in sim.cloths[0].mesh.verts])
    faces = np.array([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces])
    face_areas = np.array([[f.a] for f in sim.cloths[0].mesh.faces])
    np.save(os.path.join(conf_dir, 'ref_verts.npy'), verts)
    np.save(os.path.join(conf_dir, 'ref_faces.npy'), faces)
    np.save(os.path.join(conf_dir, 'ref_face_areas.npy'), face_areas)

def test_bag_sim():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/bag_placement/start.json'),'default_out/target',False)

def test_tshirt_xy_sim():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/clothing/start_anim.json'),'default_out/out0',False)
    for step in range(300):
        print(step)
        arcsim.sim_step()

def test_tshirt_sim():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    #arcsim.init_physics(os.path.join('conf/rigidcloth/clothing/tshirt.json'),'default_out/out0',False)
    arcsim.init_physics(os.path.join('conf/rigidcloth/clothing/tshirt_start_pinned.json'),'default_out/out0',False)
    #sim.cloths[0].materials[0].damping = torch.Tensor([0.5])
    for step in range(300):
        print(step)
        arcsim.sim_step()

def test_sysid_sim():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/sysid/start.json'),'default_out/out0',False)
    orig = sim.cloths[0].materials[0].stretching
    #sim.cloths[0].materials[0].stretching = orig*0.05
    sim.cloths[0].materials[0].stretching = orig*0.03
    for step in range(20):
        arcsim.sim_step()

def test_triangle_fold_sim():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    #arcsim.init_physics(os.path.join('conf/rigidcloth/triangle_fold/demo.json'),'default_out/out0',False)
    arcsim.init_physics(os.path.join('conf/rigidcloth/triangle_fold/start.json'),'default_out/out0',False)
    handles = [30,60]
    for step in range(30):
        sim.cloths[0].mesh.nodes[handles[0]].v += torch.Tensor([0,0,8]).double()
        sim.cloths[0].mesh.nodes[handles[1]].v += torch.Tensor([0,0,8]).double()
        arcsim.sim_step()

def test_cloth_hang_sim():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/cloth_hang/demo_fast.json'),'default_out/out0',False)
    #for step in range(500):
    #for step in range(250):
    for step in range(30):
        print(step)
        arcsim.sim_step()

def test_mask_sim():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/mask/demo.json'),'default_out/out0',False)
    for step in range(500):
        print(step)
        arcsim.sim_step()

def test_drag_demo():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/drag/drag.json'),'default_out/out0',False)
    handles = [25, 60, 30, 54]
    for step in range(20):
        for i in range(len(handles)):
            sim.cloths[0].mesh.nodes[handles[i]].v += torch.Tensor([0,0,7]).double()
        arcsim.sim_step()

if __name__ == '__main__':
    #test_drag_demo()
    #test_sysid_sim()
    #test_triangle_fold_sim()
    #test_mask_sim()
    test_cloth_hang_sim()
    #test_triangle_fold_sim()

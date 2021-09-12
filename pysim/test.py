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
    arcsim.init_physics(os.path.join('conf/rigidcloth/clothing/tshirt.json'),'default_out/out0',False)
    for step in range(30):
        print(step)
        arcsim.sim_step()

def test_sysid_sim():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/sysid/start.json'),'default_out/out0',False)
    orig_stretch = sim.cloths[0].materials[0].stretching 
    sim.cloths[0].materials[0].stretching = orig_stretch*0.01
    for step in range(20):
        arcsim.sim_step()

def test_cube_cloth():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/sysid/start_cube.json'),'default_out/out0',False)
    for node in sim.obstacles[1].curr_state_mesh.nodes:
    	node.m    *= 0.02
    	#node.m    *= 0.2
    orig_stretch = sim.cloths[0].materials[0].stretching 
    orig_damping = sim.cloths[0].materials[0].damping
    sim.cloths[0].materials[0].stretching = orig_stretch*0.15
    #sim.cloths[0].materials[0].stretching = orig_stretch
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
    for step in range(30):
        arcsim.sim_step()
    #arcsim.init_physics(os.path.join('conf/rigidcloth/cloth_hang/demo_jacket_4handles_new.json'),'default_out/out0',False)
    #for step in range(50):
    #    print(step)
    #    arcsim.sim_step()

def test_mask_sim():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/mask/demo_fast.json'),'default_out/out0',False)
    for step in range(30):
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

def test_belt_demo():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    #arcsim.init_physics(os.path.join('conf/rigidcloth/wrap/demo_fast.json'),'default_out/out0',False)
    arcsim.init_physics(os.path.join('conf/rigidcloth/wrap/demo.json'),'default_out/out0',False)
    handles = [14,15]

    for step in range(10):
        print(step)
        for i in range(len(handles)):
            sim.cloths[0].mesh.nodes[handles[i]].v += torch.Tensor([4.75,0,0.3]).double()
        arcsim.sim_step()
    for step in range(15):
        print(step)
        for i in range(len(handles)):
            sim.cloths[0].mesh.nodes[handles[i]].v += torch.Tensor([1.5,0.2,0.3]).double()
        arcsim.sim_step()
    for step in range(10):
        print(step)
        for i in range(len(handles)):
            sim.cloths[0].mesh.nodes[handles[i]].v += torch.Tensor([-2,0.5,-0.4]).double()
        arcsim.sim_step()
    for step in range(5):
        print(step)
        for i in range(len(handles)):
            sim.cloths[0].mesh.nodes[handles[i]].v += torch.Tensor([-7.5,0.4,-0.05]).double()
        arcsim.sim_step()
    for step in range(10):
        for i in range(len(handles)):
            sim.cloths[0].mesh.nodes[handles[i]].v += torch.Tensor([-1,1,-0.2]).double()
        arcsim.sim_step()


    #for step in range(10):
    #    print(step)
    #    for i in range(len(handles)):
    #        sim.cloths[0].mesh.nodes[handles[i]].v += torch.Tensor([1.0,0,0.3]).double()
    #    arcsim.sim_step()
    #for step in range(15):
    #    print(step)
    #    for i in range(len(handles)):
    #        sim.cloths[0].mesh.nodes[handles[i]].v += torch.Tensor([1.5,0.2,0.3]).double()
    #    arcsim.sim_step()
    #for step in range(10):
    #    print(step)
    #    for i in range(len(handles)):
    #        sim.cloths[0].mesh.nodes[handles[i]].v += torch.Tensor([-2,0.5,-0.4]).double()
    #    arcsim.sim_step()
    #for step in range(5):
    #    print(step)
    #    for i in range(len(handles)):
    #        sim.cloths[0].mesh.nodes[handles[i]].v += torch.Tensor([-7.5,0.4,-0.05]).double()
    #    arcsim.sim_step()

#def test_twoin_fold_demo():
#    if not os.path.exists('default_out'):
#        os.mkdir('default_out')
#    sim = arcsim.get_sim()
#    arcsim.init_physics(os.path.join('conf/rigidcloth/fold/demo_fast.json'),'default_out/out0',False)
#    #handles = [0,1,2,3]
#    #handles = [0]
#    for step in range(5):
#        arcsim.sim_step()
#    for step in range(20):
#        sim.cloths[0].mesh.nodes[0].v += torch.Tensor([7,7,1]).double()
#        sim.cloths[0].mesh.nodes[3].v += torch.Tensor([-7,-7,1]).double()
#        arcsim.sim_step()
#    for step in range(5):
#        arcsim.sim_step()

def test_twoin_fold_demo():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/fold/demo_fast.json'),'default_out/out0',False)
    for step in range(5):
        arcsim.sim_step()
    for step in range(25):
        sim.cloths[0].mesh.nodes[0].v += torch.Tensor([step/2,step/2,step/10]).double()
        sim.cloths[0].mesh.nodes[3].v += torch.Tensor([-step/2,-step/2,step/10]).double()
        #sim.cloths[0].mesh.nodes[3].v += torch.Tensor([-7,-7,1]).double()
        arcsim.sim_step()
    for step in range(5):
        arcsim.sim_step()

def test_half_fold_demo():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/fold/demo_fast.json'),'default_out/out0',False)
    orig_stretch = sim.cloths[0].materials[0].stretching 
    sim.obs_friction = torch.Tensor([1.3]).double()
    #sim.gravity *= 1.5
    #sim.cloths[0].materials[0].stretching = orig_stretch*0.04
    for step in range(20):
        print(step)
        sim.cloths[0].mesh.nodes[0].v += torch.Tensor([11.2,-6.0,8]).double()
        sim.cloths[0].mesh.nodes[2].v += torch.Tensor([11.2,+5.0,8]).double()
        arcsim.sim_step()
    for step in range(20):
        sim.cloths[0].mesh.nodes[1].v += torch.Tensor([-15.0,-6.0,8]).double()
        sim.cloths[0].mesh.nodes[3].v += torch.Tensor([-15.0,+5.0,8]).double()
        print(step)
        arcsim.sim_step()
    for step in range(30):
        print(step)
        arcsim.sim_step()
    #for step in range(10):
    #    print(step)
    #    sim.cloths[0].mesh.nodes[7].v += torch.Tensor([-8,0,5]).double()
    #    arcsim.sim_step()
    for step in range(10):
        print(step)
        sim.cloths[0].mesh.nodes[7].v += torch.Tensor([-8,0,-5]).double()
        arcsim.sim_step()
    for step in range(5):
        print(step)
        arcsim.sim_step()

def test_fricdrag_cloth_demo():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/fold/demo_fast.json'),'default_out/out0',False)
    sim.obs_friction = torch.Tensor([0.1]).double()
    for step in range(20):
        sim.cloths[0].mesh.nodes[3].v += torch.Tensor([0,10,1]).double()
        arcsim.sim_step()

def test_pants_demo():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/pants/demo.json'),'default_out/out0',False)
    for step in range(10):
        print(step)
        arcsim.sim_step()

def test_lift_cloth_corner():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/fold/demo_fast.json'),'default_out/out0',False)
    for step in range(5):
        arcsim.sim_step()
    for step in range(25):
        sim.cloths[0].mesh.nodes[0].v += torch.Tensor([0,0,step*0.6]).double()
        print(step*0.6)
        arcsim.sim_step()

def test_lasso_sim():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    #arcsim.init_physics(os.path.join('conf/rigidcloth/lasso/demo_fast.json'),'default_out/out0',False)
    #for step in range(40):
    #    print(step)
    #    arcsim.sim_step()
    arcsim.init_physics(os.path.join('conf/rigidcloth/lasso/demo_slow.json'),'default_out/out0',False)
    #print(sim.obs_friction, sim.friction)
    for step in range(50):
        print(step)
        arcsim.sim_step()

def test_materialest_sim():
    if not os.path.exists('default_out'):
        os.mkdir('default_out')
    sim = arcsim.get_sim()
    arcsim.init_physics(os.path.join('conf/rigidcloth/material_est/start.json'),'default_out/out0',False)
    print(sim.cloths[0].materials[0].stretchingori)
    print(sim.cloths[0].materials[0].bendingori)
    for step in range(40):
        print(step)
        arcsim.sim_step()

if __name__ == '__main__':
    #test_materialest_sim()
    #test_cloth_hang_sim()
    #test_mask_sim()
    #test_belt_demo()
    test_lasso_sim()
    #test_twoin_fold_demo()
    #test_cube_cloth()
    #test_fricdrag_cloth_demo()
    #test_lasso_sim()
    #test_twoin_fold_demo()
    #test_pants_demo()
    #test_half_fold_demo()
    #test_tshirt_sim()
    #test_fold_demo()
    #test_belt_demo()
    #test_sysid_sim()
    #test_lift_cloth_corner()

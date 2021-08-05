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

if __name__ == '__main__':
    get_target_triangle_fold('conf/rigidcloth/triangle_fold', 'end.json')
    #test_bag_sim()
    #print(get_target_mesh('conf/rigidcloth/half_fold', 'end.json'))

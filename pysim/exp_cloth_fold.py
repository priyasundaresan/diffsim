import torch
import arcsim
import gc
import time
import json
import sys
import gc
import os
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
now = datetime.now()
timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
#steps = 30
#epochs= 10
#steps = 45
steps = 38
epochs= 25
node_number = 0
#handles = [25, 60, 30, 54] # corners
#handles = [6,16,25,30,54,60,69,70] # side verts + 2 corners
#handles = [6,16,25,30,54,60,69,70,14,23,48] # side verts + inner side verts + 2 corners
handles = [6,16,25,30,54,60,69,70,14,23,48,11,47,74] # side verts + inner side verts + 2 corners
losses = []
param_g = torch.zeros([steps, len(handles)*3],dtype=torch.float64, requires_grad=True)
out_path = 'default_out'
os.mkdir(out_path)
with open('conf/rigidcloth/fold_starts/fold_start.json','r') as f:
    config = json.load(f)


def save_config(config, file):
    with open(file,'w') as f:
        json.dump(config, f)

save_config(config, out_path+'/conf.json')


torch.set_num_threads(16)
scalev=1

def reset_sim(sim, epoch):

    if epoch < 20:

        arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)
    else:
        arcsim.init_physics(out_path+'/conf.json',out_path+'/out',False)

def get_target_mesh():
    sim = arcsim.get_sim()
    arcsim.init_physics('conf/rigidcloth/fold_targets/half_fold.json',out_path+'/target',False)
    global node_number
    node_number = len(sim.cloths[0].mesh.nodes)
    ref = [sim.cloths[0].mesh.nodes[i].x.numpy() for i in range(node_number)]
    ref = torch.from_numpy(np.vstack(ref))
    return ref

def get_loss(sim,ref):
    reg  = torch.norm(param_g, p=2)*0.001
    loss = 0
    for i in range(ref.shape[0]):
        loss += torch.norm(ref[i]-sim.cloths[0].mesh.nodes[i].x)**2
    loss /= node_number
    loss += reg
    return loss

def run_sim(steps,sim,ref):
    # sim.obstacles[2].curr_state_mesh.dummy_node.x = param_g[1]
    print("step")
    for step in range(steps):
        print(step)
        for i in range(len(handles)):
            inc_v = param_g[step,3*i:3*i+3]
            sim.cloths[0].mesh.nodes[handles[i]].v += inc_v
            del inc_v
        arcsim.sim_step()
    loss = get_loss(sim,ref)
    return loss

#@profile
def do_train(cur_step,optimizer,scheduler,sim):
    epoch = 0
    ref = get_target_mesh()
    print(ref)
    #arcsim.delete_mesh(sim.cloths[0].mesh)
    while True:
        reset_sim(sim, epoch)
        st = time.time()
        loss = run_sim(steps, sim,ref)
        en0 = time.time()
        optimizer.zero_grad()


        loss.backward()
        en1 = time.time()
        print("=======================================")
        f.write('epoch {}:  loss={} \n'.format(epoch,  loss.data))
        print('epoch {}:  loss={} \n'.format(epoch, loss.data))

        print('forward time={}'.format(en0-st))
        print('backward time={}'.format(en1-en0))


        optimizer.step()
        print("Num cloth meshes", len(sim.cloths))
        #arcsim.delete_mesh(sim.cloths[0].mesh)
        #scheduler.step(epoch)
        losses.append(loss)
        if epoch>=epochs:
            break
        epoch = epoch + 1
        # break

def visualize_loss(losses,dir_name):
    plt.plot(losses)
    plt.title('losses')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.savefig(dir_name+'/'+'loss.jpg')

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
    tot_step = 1
    sim=arcsim.get_sim()
    # reset_sim(sim)
    lr = 10
    momentum = 0.4
    f.write('lr={} momentum={}\n'.format(lr,momentum))
    optimizer = torch.optim.SGD([{'params':param_g,'lr':lr}],momentum=momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10,2,eta_min=0.0001)
    for cur_step in range(tot_step):
        do_train(cur_step,optimizer,scheduler,sim)
    #visualize_loss(losses,default_dir)
    visualize_loss(losses,out_path)

print("done")

import bpy
import os
import sys
import json
import time
import bpy, bpy_extras
from math import *
from mathutils import *
import random
import numpy as np
from random import sample
from mathutils.bvhtree import BVHTree
import bmesh

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..'))

'''Usage: blender -b -P render.py'''
def clear_scene():
    '''Clear existing objects in scene'''
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def set_viewport_shading(mode):
    '''Makes color/texture viewable in viewport'''
    areas = bpy.context.workspace.screens[0].areas
    for area in areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = mode

def add_camera_light():
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0,0,3))
    bpy.ops.object.light_add(type='POINT', location=(0,-1,0.5))
    
    #bpy.ops.object.camera_add(location=(0,-2,0.5), rotation=(np.pi/2,0,0))
    bpy.ops.object.camera_add(location=(0,-2,1.5), rotation=(np.pi/3,0,0))
    bpy.context.scene.camera = bpy.context.object
    return bpy.context.object

def set_render_settings(engine, render_size, generate_masks=True):
    # Set rendering engine, dimensions, colorspace, images settings
    if os.path.exists("./images"):
        os.system('rm -r ./images')
    os.makedirs('./images')
    scene = bpy.context.scene
    scene.world.color = (1, 1, 1)
    scene.render.resolution_percentage = 100
    scene.render.engine = engine
    render_width, render_height = render_size
    scene.render.resolution_x = render_width
    scene.render.resolution_y = render_height
    scene.use_nodes = True
    scene.render.image_settings.file_format='JPEG'
    scene.view_settings.exposure = 1.3
    if engine == 'BLENDER_WORKBENCH':
        scene.render.image_settings.color_mode = 'RGB'
        scene.display_settings.display_device = 'None'
        scene.sequencer_colorspace_settings.name = 'XYZ'
    elif engine == "BLENDER_EEVEE":
        scene.view_settings.view_transform = 'Raw'
        scene.eevee.taa_samples = 1
        scene.eevee.taa_render_samples = 1
    elif engine == 'CYCLES':   
        scene.render.image_settings.file_format='JPEG'
        scene.cycles.samples = 20
        scene.view_settings.view_transform = 'Raw'
        scene.cycles.max_bounces = 1
        scene.cycles.min_bounces = 1
        scene.cycles.glossy_bounces = 1
        scene.cycles.transmission_bounces = 1
        scene.cycles.volume_bounces = 1
        scene.cycles.transparent_max_bounces = 1
        scene.cycles.transparent_min_bounces = 1
        scene.view_layers["View Layer"].use_pass_object_index = True
        scene.render.tile_x = 16
        scene.render.tile_y = 16

def colorize(obj, color):
    '''Add color to object'''
    mat = bpy.data.materials.new(name="Color")
    mat.use_nodes = False
    mat.diffuse_color = color
    obj.data.materials.append(mat)
    set_viewport_shading('MATERIAL')

def render(episode):
    bpy.context.scene.render.filepath = "./images/%05d.jpg"%episode
    bpy.ops.render.render(write_still=True)

def load_objs(objs_dir, episode):
    objs_fnames = [f for f in os.listdir(objs_dir) if ((f[-3:] == 'obj') and (f[:3] != 'obs') and (int(f[:4]) == episode))]
    objs = []
    COLORS = [(1,0,0,1), (0,1,0,1), (0,0,1,1)]
    for i, fname in enumerate(objs_fnames):
        color = COLORS[i]
        bpy.ops.import_scene.obj(filepath=os.path.join(objs_dir, fname))
        obj = bpy.context.selected_objects[0]
        #colorize(obj, color)
        objs.append(obj)

    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    bpy.ops.object.join()

    joined_obj = bpy.context.selected_objects[0]
    bpy.ops.object.select_all(action='DESELECT')
    joined_obj.location += Vector((-0.5,0,-0.5))
    joined_obj.rotation_euler.x -= np.pi/2
    joined_obj.location += Vector((0,-0.5,1))

    return joined_obj

def render_rollout(exp_dir, epoch_dir, steps=35, offset=0):

    objs_dir = os.path.abspath(os.path.join('..', exp_dir, epoch_dir)) 
    
    for episode in range(steps):
        obj = load_objs(objs_dir, episode)
        render(episode+offset)
        obj.select_set(True)
        bpy.ops.object.delete()
    

if __name__ == '__main__':
    clear_scene()
    camera = add_camera_light()
    render_size = (640,640)
    set_render_settings('BLENDER_EEVEE', render_size)
    offset = 0
    for i in range(0,21,5):
        render_rollout('default_out', 'out%d'%i, 20, offset=offset)
        offset += 20

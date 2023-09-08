import bpy
import numpy as np
import open3d as o3d
import os
import tqdm
from PIL import Image

from kernel.logging_utility import log_report
from kernel.geometry import _pose2Rotation, _rotation2Pose
from kernel.scale import _parseImagesFile, _parsePoints3D, _scaleFordepth, _calculateDepth
from kernel.utility import _transstring2trans,  _trans2transstring


def _is_progresslabeller_object(obj):
    if "type" in obj:
        return True
    else:
        return False

def _is_in_blender(name):
    if name in bpy.data.objects:
        return True
    else:
        log_report(
            "Error", "Object {0} is in Blender".format(name), None
        )  
        return False

def _get_workspace_name(obj):
    if _is_progresslabeller_object(obj):
        return obj.name.split(":")[0]
    else:
        return None

def _get_configuration(obj):
    if _is_progresslabeller_object(obj):
        workspaceName = _get_workspace_name(obj)
        config_id = bpy.data.objects[workspaceName + ":Setting"]['config_id']
        return config_id, bpy.context.scene.configuration[config_id]
    else:
        return None, None

def _get_reconstruction_insameworkspace(obj):
    if _is_progresslabeller_object(obj):
        workspaceName = _get_workspace_name(obj)
        if _is_in_blender(workspaceName + ":reconstruction_depthfusion"):
            return bpy.data.objects[workspaceName + ":reconstruction_depthfusion"]
        elif _is_in_blender(workspaceName + ":reconstruction"):
            return bpy.data.objects[workspaceName + ":reconstruction"]
        else:
            return None
    else:
        return None

def _is_obj_type(obj, types):
    if _is_progresslabeller_object(obj) and obj["type"] in types:
        return True
    else:
        return False

def _get_obj_insameworkspace(obj, types):
    if _is_progresslabeller_object(obj):
        obj_list = []
        workspaceName = _get_workspace_name(obj)
        for o in bpy.data.objects:
            if "type" in o and o.name.split(":")[0] == workspaceName and o["type"] in types:
                obj_list.append(o)
        return obj_list
    else:
        return []

def _apply_trans2obj(obj, trans):
    origin_pose = [list(obj.location), list(obj.rotation_quaternion)]
    origin_trans = _pose2Rotation(origin_pose)
    after_align_trans = trans.dot(origin_trans)
    after_align_pose = _rotation2Pose(after_align_trans)
    obj.location = after_align_pose[0]
    obj.rotation_quaternion = after_align_pose[1]/np.linalg.norm(after_align_pose[1])

def _get_allrgb_insameworkspace(config):
    im_list = []
    workspace_name = config.projectname
    for im in bpy.data.images:
        if im.name.split(":")[0] == workspace_name:
            im_list.append(im)
    return im_list

def _clear_allrgbdcam_insameworkspace(config):
    workspace_name = config.projectname
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if _is_obj_type(obj, "camera") and _get_workspace_name(obj) == workspace_name:
            obj.select_set(True)
    bpy.ops.object.delete() 
    for im in bpy.data.images:
        if im.name.split(":")[0] == workspace_name:
            bpy.data.images.remove(im)

def clear_initial_object():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in ["Camera", "Cube", "Light"]:
        if obj in bpy.data.objects:
            bpy.data.objects[obj].select_set(True)
    bpy.ops.object.delete()

    if "Collection" in bpy.data.collections:
        collection = bpy.data.collections.get("Collection")
        bpy.data.collections.remove(collection)
def _clear_recon_output(recon_dir):
    not_delete_file = ["label_pose.yaml", "image-list.txt"] 
    files = os.listdir(recon_dir)
    for f in files:
        p = os.path.join(recon_dir, f)
        if os.path.isfile(p) and f not in not_delete_file:
            os.remove(p)
        elif os.path.isdir(p):
            _delete(p)

def _delete(dir):
    files = os.listdir(dir)
    for f in files:
        os.remove(os.path.join(dir, f))
    os.rmdir(dir)

def _initreconpose(config):
    trans = np.identity(4)
    config.recon_trans = _trans2transstring(trans)   

def _align_reconstruction(config, scene, THRESHOLD = 0.01, NUM_THRESHOLD = 5):
    filepath = config.reconstructionsrc

    imagefilepath = os.path.join(filepath, "images.txt")
    points3Dpath = os.path.join(filepath, "points3D.txt")
    datapath = config.datasrc  

    intrinsic = np.array([
        [config.fx, 0, config.cx],
        [0, config.fy, config.cy],
        [0, 0, 1],
    ])

    depth_scale = config.depth_scale

    Camera_dict = {}
    PointsDict = {}
    PointsDepth = {}
    _parseImagesFile(imagefilepath, Camera_dict, PointsDict)
    _parsePoints3D(points3Dpath, PointsDict)
    print("Auto Aligning the scale:....")
    for camera_idx in tqdm.tqdm(Camera_dict):
            rgb_name = Camera_dict[camera_idx]["name"]
            with Image.open(os.path.join(datapath, "depth", rgb_name)) as im:
                depth = np.array(im) * depth_scale
                _scaleFordepth(depth, camera_idx, intrinsic, Camera_dict, PointsDict, PointsDepth, POSE_INVERSE = config.inverse_pose)
        
    scale = _calculateDepth(THRESHOLD = THRESHOLD, NUM_THRESHOLD = NUM_THRESHOLD, PointsDepth = PointsDepth)
    print("The scale is : {0}".format(scale))
    return scale

def _getsameinstance(config, objName): 
    workspace_name = config.projectname
    objlists = bpy.data.objects.keys()
    objNameStart = workspace_name + ":" + objName
    goalobjlist = []
    for objname in objlists:
        if objname.startswith(objNameStart):
            goalobjlist.append(objname)
    return goalobjlist

def _getnextperfixforinstance(config, objName):
    goalobjlist = _getsameinstance(config, objName)
    perfixlist = [int(name[-3:]) for name in goalobjlist]
    if len(perfixlist) == 0:
        return 1
    else:
        return (np.array(perfixlist)).max() + 1


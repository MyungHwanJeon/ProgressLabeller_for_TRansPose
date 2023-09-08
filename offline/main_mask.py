import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyrender
import trimesh
import numpy as np
from kernel.geometry import _pose2Rotation, _rotation2Pose
import cv2
from PIL import Image
import time
os.environ['PYOPENGL_PLATFORM'] = 'egl'

if __name__ == "__main__":
    config_path = sys.argv[1]
    output_dir = sys.argv[2]

    f = open(config_path)
    config = json.load(f)

    #print(config)
    
    if str(config["camera"]["name"][0]) == "R":
    
        r = pyrender.OffscreenRenderer(640, 480)
        scene = pyrender.Scene()
        cam_intrinsic = np.array([
                    [603.77647837,   0.,         329.25940456],
                    [  0.,         604.63162524, 246.48479807],
                    [  0. ,          0. ,          1.        ]], dtype=np.float64)      

        cam = pyrender.camera.IntrinsicsCamera(cam_intrinsic[0, 0],
	                               cam_intrinsic[1, 1], 
	                               cam_intrinsic[0, 2], 
	                               cam_intrinsic[1, 2], 
	                               znear=0.05, zfar=100.0, name=None)
    
    if str(config["camera"]["name"][0]) == "L":
    
        r = pyrender.OffscreenRenderer(640, 480)
        scene = pyrender.Scene()
        cam_intrinsic = np.array([
                    [609.0447980649861, 0.0, 326.84169183471016],
                    [0.0, 609.546006727566, 241.56572826467618],
                    [0, 0, 1]], dtype=np.float64)        

        cam = pyrender.camera.IntrinsicsCamera(cam_intrinsic[0, 0],
	                               cam_intrinsic[1, 1], 
	                               cam_intrinsic[0, 2], 
	                               cam_intrinsic[1, 2], 
	                               znear=0.05, zfar=100.0, name=None)            

    if str(config["camera"]["name"][0]) == "T":
    
        r = pyrender.OffscreenRenderer(640, 512)
        scene = pyrender.Scene()
        cam_intrinsic = np.array([
                    [441.62619164519106, 0.0, 318.41319462358956],
                    [0.0, 441.46556074040666, 258.6163562473],
                    [0, 0, 1]], dtype=np.float64)      

        cam = pyrender.camera.IntrinsicsCamera(cam_intrinsic[0, 0],
	                               cam_intrinsic[1, 1], 
	                               cam_intrinsic[0, 2], 
	                               cam_intrinsic[1, 2], 
	                               znear=0.05, zfar=100.0, name=None)   


    #r = pyrender.OffscreenRenderer(config["camera"]["resolution"][0], config["camera"]["resolution"][1])
    #scene = pyrender.Scene()
    #cam_intrinsic = np.array(config["camera"]["intrinsic"])

    #cam = pyrender.camera.IntrinsicsCamera(cam_intrinsic[0, 0],
    #                               cam_intrinsic[1, 1], 
    #                               cam_intrinsic[0, 2], 
    #                               cam_intrinsic[1, 2], 
    #                               znear=0.05, zfar=100.0, name=None)

    axis_align = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]]
                                )
       
    cam_pose = [list(config["camera"]["translation"]), list(config["camera"]["quaternion"])]
    cam_T = _pose2Rotation(cam_pose)
    #cam_T = cam_T.dot(axis_align)

    mesh_pose = [list(config["object"]["translation"]), list(config["object"]["quaternion"])]
    mesh_T = _pose2Rotation(mesh_pose)

    node_cam = pyrender.Node(camera=cam, matrix=cam_T)
    scene.add_node(node_cam)


    obj_mesh = trimesh.load(config["object"]["path"], force = 'mesh')
    mesh = pyrender.Mesh.from_trimesh(obj_mesh, smooth = False)
    mesh_node = pyrender.Node(mesh=mesh, matrix=mesh_T)
    scene.add_node(mesh_node)
    mesh_node.mesh.is_visible = True


    flags = pyrender.constants.RenderFlags.DEPTH_ONLY
    depth = r.render(scene, flags = flags)
    segment = depth == 0    
    
    #print(config["camera"]["name"])
    #print(config["camera"]["name"][0])
    #print(config["camera"]["name"][1:])
    rgb_img_path = config["environment"]["datasrc"] + str(config["camera"]["name"][0]) + "/undistort/" + str(config["camera"]["name"][1:]) + ".png"
    rgb_img = np.array(Image.open(rgb_img_path))
    rgb_img[segment] = 0

    output_path = output_dir + "/" + config["object"]["name"] + "/mask/"
    os.makedirs(output_path, exist_ok=True)  

    img = Image.fromarray(rgb_img)
    img.save(output_path + str(config["camera"]["name"]) + ".png")

    






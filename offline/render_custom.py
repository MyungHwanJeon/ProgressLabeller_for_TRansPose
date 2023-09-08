from cv2 import INTER_NEAREST
import pyrender
import numpy as np
import os
import json
import trimesh
import pyrender
from kernel.geometry import _pose2Rotation
from PIL import Image
from tqdm import tqdm
from scipy.io import savemat
import copy
import open3d as o3d
import cv2
import time
os.environ['PYOPENGL_PLATFORM'] = 'egl'


class offlineRender:
    def __init__(self, param, outputdir) -> None:
        print("Start offline rendering")
        self.param = param
        self.outputpath = outputdir
        self.modelsrc = self.param.modelsrc
        self.reconstructionsrc = self.param.reconstructionsrc
        self.datasrc = self.param.datasrc
        self.objects = self.param.objs
        
        self._parsecamfile()
        
        ###############################################
        
        self.renderer_R = pyrender.OffscreenRenderer(640, 480)
        self.objectmap_R = {}
        object_index = 0
        self.scene_R = pyrender.Scene()
        cam = pyrender.camera.IntrinsicsCamera(603.77647837,
                                               604.63162524, 
                                               329.25940456, 
                                               246.48479807, 
                                               znear=0.01, zfar=100.0, name=None)
        self.cam_node_R = pyrender.Node(camera=cam, matrix=np.eye(4))
        self.scene_R.add_node(self.cam_node_R)
        for obj_instancename in self.objects:
            obj = obj_instancename.split(".")[0]
            ## for full model
            if self.objects[obj_instancename]['type'] == 'normal':
                tm = trimesh.load(os.path.join(self.modelsrc, obj, obj+".obj"), force="mesh")
                mesh = pyrender.Mesh.from_trimesh(tm, smooth = False)
                node = pyrender.Node(mesh=mesh, matrix=self.objects[obj_instancename]['trans'])
                self.objectmap_R[node] = {"index":object_index, "name":obj_instancename, "trans":self.objects[obj_instancename]['trans']}
                self.scene_R.add_node(node)
                object_index += 1                                                       
        
        ###############################################
        
        self.renderer_L = pyrender.OffscreenRenderer(640, 480)
        self.objectmap_L = {}
        object_index = 0
        self.scene_L = pyrender.Scene()
        cam = pyrender.camera.IntrinsicsCamera(609.04479806,
                                               609.54600673, 
                                               326.84169183, 
                                               241.56572826, 
                                               znear=0.01, zfar=100.0, name=None)
        self.cam_node_L = pyrender.Node(camera=cam, matrix=np.eye(4))
        self.scene_L.add_node(self.cam_node_L)
        for obj_instancename in self.objects:
            obj = obj_instancename.split(".")[0]
            ## for full model
            if self.objects[obj_instancename]['type'] == 'normal':
                tm = trimesh.load(os.path.join(self.modelsrc, obj, obj+".obj"), force="mesh")
                mesh = pyrender.Mesh.from_trimesh(tm, smooth = False)
                node = pyrender.Node(mesh=mesh, matrix=self.objects[obj_instancename]['trans'])
                self.objectmap_L[node] = {"index":object_index, "name":obj_instancename, "trans":self.objects[obj_instancename]['trans']}
                self.scene_L.add_node(node)
                object_index += 1             
                
        ###############################################                
        
        self.renderer_T = pyrender.OffscreenRenderer(640, 512)
        self.objectmap_T = {}
        object_index = 0
        self.scene_T = pyrender.Scene()
        cam = pyrender.camera.IntrinsicsCamera(441.62619165,
                                               441.46556074, 
                                               318.41319462, 
                                               258.61635625, 
                                               znear=0.01, zfar=100.0, name=None)
        self.cam_node_T = pyrender.Node(camera=cam, matrix=np.eye(4))
        self.scene_T.add_node(self.cam_node_T)
        for obj_instancename in self.objects:
            obj = obj_instancename.split(".")[0]
            ## for full model
            if self.objects[obj_instancename]['type'] == 'normal':
                tm = trimesh.load(os.path.join(self.modelsrc, obj, obj+".obj"), force="mesh")
                mesh = pyrender.Mesh.from_trimesh(tm, smooth = False)
                node = pyrender.Node(mesh=mesh, matrix=self.objects[obj_instancename]['trans'])
                self.objectmap_T[node] = {"index":object_index, "name":obj_instancename, "trans":self.objects[obj_instancename]['trans']}
                self.scene_T.add_node(node)
                object_index += 1           
        
        self._createallpkgs()
        self.renderAll()

    def data_export(self, target_dir):
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)       

    def _parsecamfile(self):
        self.camposes_R = {}
        #f = open(os.path.join(self.reconstructionsrc, "campose_all_{0}.txt".format(self.interpolation_type)))
        f = open(os.path.join(self.reconstructionsrc, "campose_R.txt"))
        lines = f.readlines()
        for l in lines:
            datas = l.split(" ")
            if datas[0].isnumeric():
                self.camposes_R[datas[-1].split("\n")[0]] = _pose2Rotation([[float(datas[5]), float(datas[6]), float(datas[7])],\
                                                           [float(datas[1]), float(datas[2]), float(datas[3]), float(datas[4])]])    
        self.camposes_L = {}                                                   
        f = open(os.path.join(self.reconstructionsrc, "campose_L.txt"))
        lines = f.readlines()
        for l in lines:
            datas = l.split(" ")
            if datas[0].isnumeric():
                self.camposes_L[datas[-1].split("\n")[0]] = _pose2Rotation([[float(datas[5]), float(datas[6]), float(datas[7])],\
                                                           [float(datas[1]), float(datas[2]), float(datas[3]), float(datas[4])]]) 
        
        self.camposes_T = {}                                                      
        f = open(os.path.join(self.reconstructionsrc, "campose_T.txt"))
        lines = f.readlines()
        for l in lines:
            datas = l.split(" ")
            if datas[0].isnumeric():
                self.camposes_T[datas[-1].split("\n")[0]] = _pose2Rotation([[float(datas[5]), float(datas[6]), float(datas[7])],\
                                                           [float(datas[1]), float(datas[2]), float(datas[3]), float(datas[4])]]) 
                                                           
                                                                                                                                                                                 
    def _applytrans2cam(self):
        Axis_align = np.array([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1],]
            )
        scale = self.param.recon["scale"]
        trans = self.param.recon["trans"]
        for cam in self.camposes:
            origin_pose = self.camposes[cam]
            origin_pose[:3, 3] = origin_pose[:3, 3] * scale
            if self.CAM_INVERSE:
                origin_pose = np.linalg.inv(origin_pose).dot(Axis_align)
            else:
                origin_pose = origin_pose.dot(Axis_align)
            self.camposes[cam] = trans.dot(origin_pose)

    def _render(self, renderer, cam_node, cam_pose, object_map, scene, resolution):
        ##segimg is the instance segmentation for each part(normal or each part for the split)
        scene.set_pose(cam_node, pose=cam_pose)
        flags = pyrender.constants.RenderFlags.DEPTH_ONLY
        segimg = np.zeros((resolution[1], resolution[0]), dtype=np.uint8)

        full_depth = renderer.render(scene, flags = flags)
        for node in object_map:
            node.mesh.is_visible = False
        
        for node in object_map:
            node.mesh.is_visible = True
            depth = renderer.render(scene, flags = flags)
            mask = np.logical_and(
                (np.abs(depth - full_depth) < 1e-6), np.abs(full_depth) > 0
            )
            segimg[mask] = object_map[node]['index'] + 1
            node.mesh.is_visible = False
        
        for node in object_map:
            node.mesh.is_visible = True
        return segimg

    def _createpkg(self, dir):
        if os.path.exists(dir):
            return True
        else:
            if self._createpkg(os.path.dirname(dir)):
                os.mkdir(dir)
                return self._createpkg(os.path.dirname(dir))

    def _createallpkgs(self):
        for node in self.objectmap_R:            
            self._createpkg(os.path.join(self.outputpath, self.objectmap_R[node]["name"], "R"))
            self._createpkg(os.path.join(self.outputpath, self.objectmap_R[node]["name"], "R/pose"))
            self._createpkg(os.path.join(self.outputpath, self.objectmap_R[node]["name"], "R/mask"))
            self._createpkg(os.path.join(self.outputpath, self.objectmap_R[node]["name"], "R/depth"))   
            
        for node in self.objectmap_L:    
            self._createpkg(os.path.join(self.outputpath, self.objectmap_L[node]["name"], "L"))
            self._createpkg(os.path.join(self.outputpath, self.objectmap_L[node]["name"], "L/pose"))
            self._createpkg(os.path.join(self.outputpath, self.objectmap_L[node]["name"], "L/mask"))
            
        for node in self.objectmap_T:      
            self._createpkg(os.path.join(self.outputpath, self.objectmap_T[node]["name"], "T"))
            self._createpkg(os.path.join(self.outputpath, self.objectmap_T[node]["name"], "T/pose"))
            self._createpkg(os.path.join(self.outputpath, self.objectmap_T[node]["name"], "T/mask"))
            
    
    def renderAll(self):
        ## generate whole output dataset
        Axis_align = np.array([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1],]
                                )
        
        print("Start offline rendering - Right Cam")
        for cam in tqdm(self.camposes_R):
            camT = self.camposes_R[cam].dot(Axis_align)
            #renderer, cam_node, cam_pose, object_map, scene, resolution
            segment_R = self._render(self.renderer_R, self.cam_node_R, camT, self.objectmap_R, self.scene_R, [640, 480])
            perfix = cam.split(".")[0]
            input_R = np.array(Image.open(os.path.join(self.datasrc, "R/undistort", cam)))
            #input_R = np.array(Image.open(os.path.join(self.datasrc, "R/rgb", cam)))
            
            self.scene_R.set_pose(self.cam_node_R, pose=camT)
            full_depth = self.renderer_R.render(self.scene_R, flags = pyrender.constants.RenderFlags.DEPTH_ONLY) *1000
            full_depth = full_depth.astype(np.uint16)       
            input_depth_R = cv2.imread(os.path.join(self.datasrc, "R/depth_undistort", cam), cv2.IMREAD_UNCHANGED)    
            #input_depth_R = cv2.imread(os.path.join(self.datasrc, "R/depth", cam), cv2.IMREAD_UNCHANGED)    
            
            input_depth_R[full_depth != 0] = full_depth[full_depth != 0]       
            	
            	

            for node in self.objectmap_R:
                posepath = os.path.join(self.outputpath, self.objectmap_R[node]["name"], "R/pose")
                R_path = os.path.join(self.outputpath, self.objectmap_R[node]["name"], "R/mask")
                modelT = self.objectmap_R[node]["trans"]
                model_camT = np.linalg.inv(modelT).dot(self.camposes_R[cam])
                self._createpose(posepath, perfix, model_camT)
                self._createrbg(input_R, segment_R, os.path.join(R_path, cam), self.objectmap_R[node]["index"] + 1)
                
                cv2.imwrite(os.path.join(self.outputpath, self.objectmap_R[node]["name"], "R/depth", cam), input_depth_R)
                
                
        print("Start offline rendering - Left Cam")        
        for cam in tqdm(self.camposes_L):
            camT = self.camposes_L[cam].dot(Axis_align)
            #renderer, cam_node, cam_pose, object_map, scene, resolution
            segment_L = self._render(self.renderer_L, self.cam_node_L, camT, self.objectmap_L, self.scene_L, [640, 480])
            perfix = cam.split(".")[0]
            input_L = np.array(Image.open(os.path.join(self.datasrc, "L/undistort", cam)))

            for node in self.objectmap_L:
                posepath = os.path.join(self.outputpath, self.objectmap_L[node]["name"], "L/pose")
                R_path = os.path.join(self.outputpath, self.objectmap_L[node]["name"], "L/mask")
                modelT = self.objectmap_L[node]["trans"]
                model_camT = np.linalg.inv(modelT).dot(self.camposes_L[cam])
                self._createpose(posepath, perfix, model_camT)
                self._createrbg(input_L, segment_L, os.path.join(R_path, cam), self.objectmap_L[node]["index"] + 1)
                
        

        print("Start offline rendering - Thermal Cam")
        for cam in tqdm(self.camposes_T):
            camT = self.camposes_T[cam].dot(Axis_align)
            #renderer, cam_node, cam_pose, object_map, scene, resolution
            segment_T = self._render(self.renderer_T, self.cam_node_T, camT, self.objectmap_T, self.scene_T, [640, 512])
            perfix = cam.split(".")[0]
            input_T = np.array(Image.open(os.path.join(self.datasrc, "T/undistort", cam)))

            for node in self.objectmap_T:
                posepath = os.path.join(self.outputpath, self.objectmap_T[node]["name"], "T/pose")
                R_path = os.path.join(self.outputpath, self.objectmap_T[node]["name"], "T/mask")
                modelT = self.objectmap_T[node]["trans"]
                model_camT = np.linalg.inv(modelT).dot(self.camposes_T[cam])
                self._createpose(posepath, perfix, model_camT)
                self._createrbg_thermal(input_T, segment_T, os.path.join(R_path, cam), self.objectmap_T[node]["index"] + 1)        
                
                
                
                
                
                                                          

    def _createpose(self, path, perfix, T):
        posefileName = os.path.join(path, perfix + ".txt")
        # np.savetxt(posefileName, np.linalg.inv(T), fmt='%f', delimiter=' ')
        np.savetxt(posefileName, T, fmt='%f', delimiter=' ')

    def _createrbg(self, inputrgb, segment, outputpath, segment_index):
        rgb = inputrgb.copy()
        mask = np.repeat((segment != segment_index)[:, :, np.newaxis], 3, axis=2)
        rgb[mask] = 0
        img = Image.fromarray(rgb)
        img.save(outputpath)    
        
        
    def _createrbg_thermal(self, inputrgb, segment, outputpath, segment_index):
        rgb = inputrgb.copy()
        
        h, w = rgb.shape
        min_thresh = np.min(rgb)
        max_thresh = np.max(rgb)

        if (min_thresh != None) and (max_thresh != None):
            for i in range(h):
                for j in range(w):
                    if (rgb[i][j]) > max_thresh:
                        rgb[i][j] = max_thresh

                    elif (rgb[i][j]) < min_thresh:
                        rgb[i][j] = min_thresh

        normalized_array = (rgb - min_thresh) / (max_thresh - min_thresh)

        norm_img = (normalized_array * 255).astype(np.uint8)
        
        mask = np.repeat((segment != segment_index)[:, :, np.newaxis], 3, axis=2)
        norm_img[mask[:, :, 0]] = 0
        img = Image.fromarray(norm_img)
        img.save(outputpath)          
        
        
    
    def _getbbx(self, mask):
        pixel_list = np.where(mask)
        if np.any(pixel_list):
            top = pixel_list[0].min()
            bottom = pixel_list[0].max()
            left = pixel_list[1].min()
            right = pixel_list[1].max()
            return True, [int(left), int(top), int(right - left), int(bottom - top)]
        else:
            return False, []

    def _getbbxycb(self, mask):
        pixel_list = np.where(mask)
        if np.any(pixel_list):
            top = pixel_list[0].min()
            bottom = pixel_list[0].max()
            left = pixel_list[1].min()
            right = pixel_list[1].max()
            return True, [int(left), int(top), int(right), int(bottom)]
        else:
            return False, []           

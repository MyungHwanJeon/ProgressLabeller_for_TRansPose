import os
import sys
import pyglet
os.environ["PYOPENGL_PLATFORM"] = "egl"
pyglet.options['shadow_window'] = False

import numpy as np
import time
import glob
import random
import sys
import cv2
import json

import torch 
import torch.nn as nn
from torchvision.ops import RoIPool
import pytorch3d
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    Textures,
    TexturesUV,
    TexturesVertex,
    PerspectiveCameras,
    MeshRendererWithFragments,
    SoftSilhouetteShader,
    TexturesAtlas,
    TexturesUV,
    BlendParams
)

class Model(nn.Module):
    def __init__(self, mesh, AA_World2Object, t_World2Object):  
        super().__init__()
        self.mesh = mesh
        self.device = mesh.device
                
        self.AA_World2Object = nn.Parameter(AA_World2Object.to(self.device))
        #self.AA_World2Object = AA_World2Object.to(self.device)
        self.t_World2Object = nn.Parameter(t_World2Object.to(self.device))
        #self.t_World2Object = t_World2Object.to(self.device)
               
        self.keys = ["R", "L", "T"]       
               
    def forward(self, renderers, gt_masks, R_World2Cams, t_World2Cams):
        
        loss_all = 0
        for k in renderers.keys():
            for i in range(len(gt_masks[k])):
              
                R_World2Object = pytorch3d.transforms.axis_angle_to_matrix(self.AA_World2Object)  
                R_Cam2Object = torch.bmm(R_World2Cams[k][i].inverse().view(1, 3, 3), R_World2Object.view(1, 3, 3))
        
                R_Cam2Object_render = R_Cam2Object.permute(0, 2, 1)
                R_Cam2Object_render[:, :, :2] *= -1    
                
                r1t2 = torch.bmm(R_World2Cams[k][i].inverse().view(1, 3, 3), self.t_World2Object.view(1, 3, 1))
                r1t1 = torch.bmm(R_World2Cams[k][i].inverse().view(1, 3, 3), t_World2Cams[k][i].view(1, 3, 1))
                t_Cam2Object_render = r1t2-r1t1
                t_Cam2Object_render[:, :2] *= -1   
                
                obj_mask = renderers[k](meshes_world=self.mesh, R=R_Cam2Object_render.view(1, 3, 3), T=t_Cam2Object_render.view(1, 3)*torch.tensor(1000.))                     
                
                pred_flat = torch.flatten(obj_mask[:, :, :, 3]/(obj_mask[:, :, :, 3] + 1e-4), start_dim=1)  
                gt_flat = torch.flatten(torch.from_numpy(gt_masks[k][i]).to(self.device).view(1, gt_masks[k][i].shape[0], gt_masks[k][i].shape[1]), start_dim=1)   
                loss = nn.MSELoss(reduction='none')(pred_flat, gt_flat)
                values, indices = torch.topk(loss, int(loss.shape[1]), dim=1)       
                loss = torch.sum(values)        
                loss_all += loss       
   
        return loss_all        



if __name__ == "__main__":

    config_path = sys.argv[1]
    output_dir = sys.argv[2]

    f = open(config_path)
    config = json.load(f)
    
    #print(config)

    device = "cuda:0"
    Deg2Rad = torch.tensor(np.pi / 180.).to(device)
     
    verts, faces, aux = load_obj(config["object"]["path"])
    verts = verts * 1000.                         
               
    tex_maps = aux.texture_images
    if tex_maps is not None and len(tex_maps) > 0:
        verts_uvs = aux.verts_uvs.to(device) 
        faces_uvs = faces.textures_idx.to(device)
        image = list(tex_maps.values())[0].to(device)[None]
        tex = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image)
        obj_mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex)
    else:
        obj_mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)])    

    ####################################################################################################################################################################################
        
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    raster_settings_R = RasterizationSettings(image_size=[480, 640], blur_radius=0.000, faces_per_pixel=1, bin_size=None)   
    cameras_R = PerspectiveCameras(
                            focal_length=np.array([603.77647837, 604.63162524]).reshape((1, 2)), 
                            principal_point=np.array([329.25940456, 246.48479807]).reshape((1, 2)), 
                            image_size=np.array([480, 640]).reshape((1, 2)), 
                            device=device,
                            in_ndc=False                            
                            )                        
    renderer_R = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras_R,
            raster_settings=raster_settings_R
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    cameras_R.zfar = 10000
    cameras_R.znear = 1  
        
    raster_settings_L = RasterizationSettings(image_size=[480, 640], blur_radius=0.000, faces_per_pixel=1, bin_size=None)       
    cameras_L = PerspectiveCameras(
                            focal_length=np.array([609.0447980649861, 609.546006727566]).reshape((1, 2)), 
                            principal_point=np.array([326.84169183471016, 241.56572826467618]).reshape((1, 2)), 
                            image_size=np.array([480, 640]).reshape((1, 2)), 
                            device=device,
                            in_ndc=False                            
                            )                        
    renderer_L = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras_L,
            raster_settings=raster_settings_L
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    cameras_L.zfar = 10000
    cameras_L.znear = 1      
        

    raster_settings_T = RasterizationSettings(image_size=[512, 640], blur_radius=0.000, faces_per_pixel=1, bin_size=None)   
    cameras_T = PerspectiveCameras(
                            focal_length=np.array([441.62619164519106, 441.46556074040666]).reshape((1, 2)), 
                            principal_point=np.array([318.41319462358956, 258.6163562473]).reshape((1, 2)), 
                            image_size=np.array([512, 640]).reshape((1, 2)), 
                            device=device,
                            in_ndc=False                            
                            )                        
    renderer_T = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras_T,
            raster_settings=raster_settings_T
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    cameras_T.zfar = 10000
    cameras_T.znear = 1          
      
    renderers = {"R":renderer_R, "L":renderer_L, "T":renderer_T}
    
    ####################################################################################################################################################################################    	   
    
    file_list_idx = {"R":list(), "L":list(), "T":list()}
    file_list = os.listdir(output_dir + "/" + config["object"]["name"] + "/mask/") 
    n_pose = len(file_list)
    for i in range(len(file_list)):
    
        cam_idx = file_list[i].split(".")[0][0]    
        img_idx = file_list[i].split(".")[0][1:]        
        file_list_idx[str(cam_idx)].append(int(img_idx))


    ####################################################################################################################################################################################    	   

    Q_World2Cams = {"R":torch.zeros([len(file_list_idx["R"]), 4], dtype=torch.float32).to(device),
                    "L":torch.zeros([len(file_list_idx["L"]), 4], dtype=torch.float32).to(device),
                    "T":torch.zeros([len(file_list_idx["T"]), 4], dtype=torch.float32).to(device)}
    AA_World2Cams = {"R":torch.zeros([len(file_list_idx["R"]), 3], dtype=torch.float32).to(device),
                    "L":torch.zeros([len(file_list_idx["L"]), 3], dtype=torch.float32).to(device),
                    "T":torch.zeros([len(file_list_idx["T"]), 3], dtype=torch.float32).to(device)} 
    R_World2Cams = {"R":torch.zeros([len(file_list_idx["R"]), 3, 3], dtype=torch.float32).to(device),
                    "L":torch.zeros([len(file_list_idx["L"]), 3, 3], dtype=torch.float32).to(device),
                    "T":torch.zeros([len(file_list_idx["T"]), 3, 3], dtype=torch.float32).to(device)}                          
    t_World2Cams = {"R":torch.zeros([len(file_list_idx["R"]), 3], dtype=torch.float32).to(device),
                    "L":torch.zeros([len(file_list_idx["L"]), 3], dtype=torch.float32).to(device),
                    "T":torch.zeros([len(file_list_idx["T"]), 3], dtype=torch.float32).to(device)}      
    T_World2Cams = {"R":torch.zeros([len(file_list_idx["R"]), 4, 4], dtype=torch.float32).to(device),
                    "L":torch.zeros([len(file_list_idx["L"]), 4, 4], dtype=torch.float32).to(device),
                    "T":torch.zeros([len(file_list_idx["T"]), 4, 4], dtype=torch.float32).to(device)}                                                       

    for k in file_list_idx.keys():
    
        f = open(config["environment"]["reconstructionsrc"] + "/campose_" + k + ".txt", 'r')
        lines = f.readlines() 
        f.close()    
    
        for i in range(len(file_list_idx[k])):
        
            token = lines[file_list_idx[k][i] + 1].split(" ")
        
            Q_World2Cams[k][i, 0] = float(token[1])
            Q_World2Cams[k][i, 1] = float(token[2])
            Q_World2Cams[k][i, 2] = float(token[3])
            Q_World2Cams[k][i, 3] = float(token[4])        
            R_World2Cams[k][i]  = pytorch3d.transforms.quaternion_to_matrix(Q_World2Cams[k][i]).to(device)
            AA_World2Cams[k][i] = pytorch3d.transforms.quaternion_to_axis_angle(Q_World2Cams[k][i]).to(device)
        
            t_World2Cams[k][i, 0] = float(token[5])# * 1000.
            t_World2Cams[k][i, 1] = float(token[6])# * 1000.
            t_World2Cams[k][i, 2] = float(token[7])# * 1000.
        
            T_World2Cams[k][i, :3, :4]  = torch.cat((R_World2Cams[k][i].reshape(-1, 3, 3), t_World2Cams[k][i].reshape(-1, 3, 1)), 2)
            T_World2Cams[k][i, 3, 3] = 1.
                     
    ####################################################################################################################################################################################    	   

    gt_masks = {"R":np.zeros([len(file_list_idx["R"]), 480, 640], dtype=np.float32),
                "L":np.zeros([len(file_list_idx["L"]), 480, 640], dtype=np.float32),
                "T":np.zeros([len(file_list_idx["T"]), 512, 640], dtype=np.float32)
               }
               
    for k in file_list_idx.keys():               
        for i in range(len(file_list_idx[k])):
            
            gt_mask = cv2.imread(output_dir + "/" + config["object"]["name"] + "/mask/" + k + "%06d.png"%file_list_idx[k][i])
            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)  
            
            #cv2.imshow(k + "%06d.png"%file_list_idx[k][i], gt_mask)
            #cv2.waitKey(0)
            
            gt_masks[k][i] = (gt_mask != 0)# * 0.5
            
    ####################################################################################################################################################################################        
        
    Q_World2Object = torch.tensor([config["object"]["quaternion"][0],  config["object"]["quaternion"][1], config["object"]["quaternion"][2], config["object"]["quaternion"][3]]).reshape(-1, 4).to(device)  # w x y z
    R_World2Object = pytorch3d.transforms.quaternion_to_matrix(Q_World2Object).to(device)
    AA_World2Object = pytorch3d.transforms.quaternion_to_axis_angle(Q_World2Object).to(device)
    t_World2Object = torch.tensor([config["object"]["translation"][0],  config["object"]["translation"][1], config["object"]["translation"][2]]).reshape(-1, 3).to(device) # x y z
    T_World2Object = torch.cat((R_World2Object, t_World2Object.reshape(-1, 3, 1).to(device)), 2)
    T_World2Object = torch.cat((T_World2Object, torch.tensor([0., 0., 0., 1.]).reshape(-1, 1, 4).to(device)), 1)        
    
    #for k in file_list_idx.keys(): 
    #    for i in range(len(file_list_idx[k])):
    #        print(file_list_idx[k][i])
        
    #    print()    
    #    print(Q_World2Cams[k])
        
    #    print()    
    #    print(R_World2Cams[k])
        
    #    print()    
    #    print(AA_World2Cams[k])
        
    #    print()    
    #    print(t_World2Cams[k])
        
    #    print()    
    #    print(T_World2Cams[k])        
                             

    epoch = 800
    model = Model(obj_mesh, AA_World2Object, t_World2Object).to(device)    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=float(config["optimization"]["translation_learning_rate"]))
    optimizer_r = torch.optim.RMSprop(model.parameters(), lr=float(config["optimization"]["rotation_learning_rate"]))
    #optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["optimization"]["translation_learning_rate"]))
    #optimizer_r = torch.optim.AdamW(model.parameters(), lr=float(config["optimization"]["rotation_learning_rate"]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 700], gamma=0.5)
    scheduler_r = torch.optim.lr_scheduler.MultiStepLR(optimizer_r, milestones=[500, 700], gamma=0.5)
    
    smallest_loss = 1000000000
    optimal_t_World2Object = 0
    optimal_AA_World2Object = 0
    
    for i in range(epoch):

        model.AA_World2Object.requires_grad = False
        model.t_World2Object.requires_grad = True        
        for kkk in range(1):
            optimizer.zero_grad()        
            loss = model(renderers, gt_masks, R_World2Cams, t_World2Cams)
            loss.backward()
            optimizer.step()   
            scheduler.step()                               
        
            
        
        model.AA_World2Object.requires_grad = True
        model.t_World2Object.requires_grad = False
        for kkk in range(1):
            optimizer.zero_grad()        
            loss = model(renderers, gt_masks, R_World2Cams, t_World2Cams)
            loss.backward()
            optimizer_r.step()   
            scheduler_r.step()

        
        loss = model(renderers, gt_masks, R_World2Cams, t_World2Cams)
        if loss < smallest_loss:
            smallest_loss = loss
            optimal_t_World2Object = model.t_World2Object.clone().detach()
            optimal_AA_World2Object = model.AA_World2Object.clone().detach()
            
            for k in renderers.keys():
                for j in range(len(gt_masks[k])):
              
                    R_World2Object = pytorch3d.transforms.axis_angle_to_matrix(model.AA_World2Object)  
                    R_Cam2Object = torch.bmm(R_World2Cams[k][j].inverse().view(1, 3, 3), R_World2Object.view(1, 3, 3))
        
                    R_Cam2Object_render = R_Cam2Object.permute(0, 2, 1)
                    R_Cam2Object_render[:, :, :2] *= -1    
                
                    r1t2 = torch.bmm(R_World2Cams[k][j].inverse().view(1, 3, 3), model.t_World2Object.view(1, 3, 1))
                    r1t1 = torch.bmm(R_World2Cams[k][j].inverse().view(1, 3, 3), t_World2Cams[k][j].view(1, 3, 1))
                    t_Cam2Object_render = r1t2-r1t1
                    t_Cam2Object_render[:, :2] *= -1   
                
                    obj_mask = renderers[k](meshes_world=obj_mesh, R=R_Cam2Object_render.view(1, 3, 3), T=t_Cam2Object_render.view(1, 3)*torch.tensor(1000.))
                                        
                    cv2.imshow(k + "%06d"%file_list_idx[k][j], cv2.addWeighted(gt_masks[k][j], 0.5, (obj_mask[0, :, :, 3]/(obj_mask[0, :, :, 3] + 1e-4)).data.cpu().numpy(), 0.5, 0.0))
                    cv2.waitKey(30)
            
        
        if loss < float(config["optimization"]["stop_criteria"] * n_pose):
            break        


         
        
        
        print()
        print("t_lr: ", optimizer.param_groups[0]['lr'], " ", "r_lr: ", optimizer_r.param_groups[0]['lr'])
        print(i, "loss:", loss.item())    
        print("t:", model.t_World2Object.data.cpu().numpy()*1000.)
        print("R:", model.AA_World2Object.data.cpu().numpy()*(180/np.pi))
    
        #if i%10 == 0:
            
        #    for k in renderers.keys():
        #        for j in range(len(gt_masks[k])):
              
        #            R_World2Object = pytorch3d.transforms.axis_angle_to_matrix(model.AA_World2Object)  
        #            R_Cam2Object = torch.bmm(R_World2Cams[k][j].inverse().view(1, 3, 3), R_World2Object.view(1, 3, 3))
        
        #            R_Cam2Object_render = R_Cam2Object.permute(0, 2, 1)
        #            R_Cam2Object_render[:, :, :2] *= -1    
                
        #            r1t2 = torch.bmm(R_World2Cams[k][j].inverse().view(1, 3, 3), model.t_World2Object.view(1, 3, 1))
        #            r1t1 = torch.bmm(R_World2Cams[k][j].inverse().view(1, 3, 3), t_World2Cams[k][j].view(1, 3, 1))
        #            t_Cam2Object_render = r1t2-r1t1
        #            t_Cam2Object_render[:, :2] *= -1   
                
        #            obj_mask = renderers[k](meshes_world=obj_mesh, R=R_Cam2Object_render.view(1, 3, 3), T=t_Cam2Object_render.view(1, 3)*torch.tensor(1000.))
                                        
        #            cv2.imshow(k + "%06d"%file_list_idx[k][j], cv2.addWeighted(gt_masks[k][j], 0.5, (obj_mask[0, :, :, 3]/(obj_mask[0, :, :, 3] + 1e-4)).data.cpu().numpy(), 0.5, 0.0))
        #            cv2.waitKey(1)
    
    
    print("============================================================")
    print("Optimization End")
    print("loss:", smallest_loss.item())   
    print("t:", optimal_t_World2Object.data.cpu().numpy()*1000.)
    print("R:", optimal_AA_World2Object.data.cpu().numpy()*(180/np.pi))
    print("============================================================")
    
    result_t = optimal_t_World2Object.data.cpu().numpy().reshape(3)  
    result_q = pytorch3d.transforms.axis_angle_to_quaternion(optimal_AA_World2Object).data.cpu().numpy().reshape(4)                      
    output_dict = {
            'projectname': config["projectname"],
            'object':{
                "name":config['object']['name'],
                "path":config['object']['path'],
                "translation":[float(result_t[0]), float(result_t[1]), float(result_t[2])],
                "quaternion":[float(result_q[0]), float(result_q[1]), float(result_q[2]), float(result_q[3])],
            }
        }

    with open("/tmp/progresslabeller_mask_result.json", "w") as f:
        json.dump(output_dict, f, indent = True)    
	                         
    
    
    


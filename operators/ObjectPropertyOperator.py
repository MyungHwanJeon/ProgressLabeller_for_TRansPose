import bpy
from bpy.props import StringProperty, EnumProperty, FloatProperty, IntProperty
from bpy_extras.view3d_utils import location_3d_to_region_2d
from bpy.types import Operator
import numpy as np
from PIL import Image
import os
import time
import open3d as o3d
import tqdm
import trimesh
import pyrender
import multiprocessing
import subprocess
import sys
import json



from kernel.render import save_img
from kernel.geometry import plane_alignment, transform_from_plane, _pose2Rotation, _rotation2Pose, modelICP, globalRegisteration
from kernel.logging_utility import log_report
from kernel.loader import load_cam_img_depth, load_reconstruction_result, updateprojectname, removeworkspace, load_pc
from kernel.blender_utility import \
    _get_configuration, _get_reconstruction_insameworkspace, _get_obj_insameworkspace, _get_workspace_name, _apply_trans2obj, \
    _align_reconstruction
from registeration.init_configuration import config
from kernel.utility import _trans2transstring,  _parse_camfile, _select_sample_files
from kernel.blender_utility import _is_progresslabeller_object, _initreconpose, _get_obj_insameworkspace
from panel.FloatScreenPanel import draw_for_area
import registeration.register 


class MaskOptimization(Operator):
    """Using RANSAC to detect the plane and align it"""
    bl_idname = "object_property.maskoptimation"  
    bl_label = "Mask Optimization"

    def execute(self, context):
        
        obj = context.object
        assert obj['type'] == 'model'
        ws_name = _get_workspace_name(obj)
        if len(bpy.context.selected_objects) > 0:
            obj_name = bpy.context.selected_objects[0].name
        
        
            setting_object = 0
            visible_objects = [obj for obj in bpy.context.view_layer.objects if obj.visible_get()]

            for obj_ in visible_objects:
                if obj_.name == "untitled:Setting":
                    setting_object = obj_
        
            config_id = setting_object['config_id']
            config = bpy.context.scene.configuration[config_id]
        

            path = setting_object['dir']
            output_filepath = (setting_object["dir"] if setting_object["dir"].endswith("/") else setting_object["dir"] + "/") + "output"
            obj_pose = np.zeros(7, dtype=np.float32)
            obj_context = 0             
        
        
            if context.selected_objects[0]["type"] == "model":

                obj_pose[0] = context.selected_objects[0].location[0]
                obj_pose[1] = context.selected_objects[0].location[1]
                obj_pose[2] = context.selected_objects[0].location[2]
            
                context.selected_objects[0].rotation_mode = 'QUATERNION'

                obj_pose[3] = context.selected_objects[0].rotation_quaternion[0]
                obj_pose[4] = context.selected_objects[0].rotation_quaternion[1]
                obj_pose[5] = context.selected_objects[0].rotation_quaternion[2]
                obj_pose[6] = context.selected_objects[0].rotation_quaternion[3]

                obj_context = context.selected_objects[0]

            else:

                log_report("Error", "error", None)
                return {'FINISHED'}        
        
        
            output_dict = {
                'projectname': config.projectname,
                'environment': {
                    'modelsrc':config.modelsrc,
                    "reconstructionsrc":config.reconstructionsrc,
                    "datasrc": config.datasrc,           
                },
                'camera':{
                    "resolution": [config.resX, config.resY],
                    "intrinsic": [[config.fx, 0, config.cx],
                                  [0, config.fy, config.cy],
                                  [0, 0, 1]],
                },
                'object':{
                    "name":obj_context.name.split(":")[1],
                    "path":obj_context["path"],
                    "translation":[float(obj_pose[0]), float(obj_pose[1]), float(obj_pose[2])],
                    "quaternion":[float(obj_pose[3]), float(obj_pose[4]), float(obj_pose[5]), float(obj_pose[6])],
                },
                'optimization':{
                    "stop_criteria":context.selected_objects[0].optimizationparas.stop_criteria,
                    "translation_learning_rate":context.selected_objects[0].optimizationparas.translation_learning_rate,
                    "rotation_learning_rate":context.selected_objects[0].optimizationparas.rotation_learning_rate
                }
            }           
            
            with open("/tmp/progresslabeller_pose_opt.json", "w") as f:
                json.dump(output_dict, f, indent = True)

            source = os.path.dirname(os.path.dirname(__file__))
            code_path = os.path.join(source, "./", "offline", "mask_opt.py")
            subprocess.call("{} {} {} {}".format(sys.executable, code_path, "/tmp/progresslabeller_pose_opt.json", output_filepath), shell=True)

            os.remove("/tmp/progresslabeller_pose_opt.json")

            f = open("/tmp/progresslabeller_mask_result.json")
            result = json.load(f)
        
            objworkspacename = result["projectname"] + ":" + result["object"]["name"]        
            bpy.data.objects[objworkspacename].location = np.array(result["object"]["translation"])
            bpy.data.objects[objworkspacename].rotation_mode = 'QUATERNION'
            bpy.data.objects[objworkspacename].rotation_quaternion = np.array(result["object"]["quaternion"]) / np.linalg.norm(np.array(result["object"]["quaternion"]))
        
            os.remove("/tmp/progresslabeller_mask_result.json")


            log_report("Info", "Pose Optimization", None)                           
        
        
        return {'FINISHED'}
        
class MaskOptimizationConfig(bpy.types.PropertyGroup):
    stop_criteria: bpy.props.FloatProperty(name="stop_criteria", 
                                        description="stop_criteria", 
                                        default=100.0, 
                                        min=0., 
                                        max=1000., 
                                        step=1, 
                                        precision=3)

    translation_learning_rate: bpy.props.FloatProperty(name="translation_learning_rate", 
                                description="translation_learning_rate", 
                                default=0.0005, 
                                min=0, 
                                max=1, 
                                step=0.0001,
                                precision=5)
    rotation_learning_rate: bpy.props.FloatProperty(name="rotation_learning_rate", 
                                        description="rotation_learning_rate", 
                                        default=0.005, 
                                        min=0, 
                                        max=1, 
                                        step=0.0001,
                                        precision=5)


class PlaneAlignment(Operator):
    """Using RANSAC to detect the plane and align it"""
    bl_idname = "object_property.planealignment"  
    bl_label = "Plane Alignment"

    def execute(self, context):
        recon = context.object
        log_report(
            "INFO", "Starting calculate the plane function", None
        )      
        
        [a, b, c, d], plane_center = plane_alignment(recon["path"], 
                                                     recon["scale"],
                                                     np.array(recon["alignT"]),
                                                     bpy.context.scene.planalignmentparas.threshold,
                                                     bpy.context.scene.planalignmentparas.n,
                                                     bpy.context.scene.planalignmentparas.iteration)
        log_report(
            "INFO", f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0", None
        )        
        if d < 0:
            trans  = transform_from_plane([-a, -b, -c, -d], plane_center)
        else:
            trans  = transform_from_plane([a, b, c, d], plane_center)
        log_report(
            "INFO", "Starting Transform the scene", None
        )   

        obj_lists = _get_obj_insameworkspace(context.object, ["reconstruction", "camera"])
        
        for obj in obj_lists:
            _apply_trans2obj(obj, trans)


        recons = _get_obj_insameworkspace(recon, ["reconstruction"])

        for obj in recons:
            print(obj.name)
            obj["alignT"] = (trans.dot(np.array(obj["alignT"]))).tolist()

        _, config = _get_configuration(context.object)
        config.recon_trans = _trans2transstring(np.array(obj["alignT"]))

        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width = 400)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.label(text="Set Plane Alignment (ICP) Parameters:")
        box = layout.box() 
        row = box.row()
        row.prop(scene.planalignmentparas, "threshold") 
        row = box.row()
        row.prop(scene.planalignmentparas, "n") 
        row = box.row()
        row.prop(scene.planalignmentparas, "iteration") 

class PlaneAlignmentConfig(bpy.types.PropertyGroup):
    threshold: bpy.props.FloatProperty(name="Inlier Threshold", 
                                        description="Inlier threshold for points aligned to plane", 
                                        default=0.01, 
                                        min=0.001, 
                                        max=1.000, 
                                        step=0.001, 
                                        precision=3)

    n: bpy.props.IntProperty(name="Fit Number", 
                                description="Numbers for fitting a plane", 
                                default=3, 
                                min=3, 
                                max=10, 
                                step=1)
    iteration: bpy.props.IntProperty(name="Fit Iteration", 
                                        description="ICP Iteration", 
                                        default=1000, 
                                        min=10, 
                                        max=10000, 
                                        step=100)


class ImportCamRGBDepth(Operator):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "object_property.importcamrgbdepth"  
    bl_label = "Import RGB & Depth"


    def execute(self, context): 
        config_id, config = _get_configuration(context.object)
        packagepath = config.datasrc
        files = os.listdir(packagepath)
        #if "rgb_undistort" not in files or "depth" not in files:
        #    log_report(
        #        "Error", "either rgb or depth package is not in the datasrc", None
        #    )       
        #elif config.resX == 0 or config.resY == 0\
        #    or config.fx == 0 or config.fy == 0\
        #    or config.cx == 0 or config.cy == 0:
        #    log_report(
        #        "ERROR", "Please set the camera parameters before loading the RGB and Camera", None
        #    )   
        #    return {'FINISHED'}      
        #else:   
        #    load_cam_img_depth(packagepath, config_id, camera_display_scale = 0.1, sample_rate=config.sample_rate)
        #return {'FINISHED'}
    
    def invoke(self, context, event):
        config_id, config = _get_configuration(context.object)
        #files = os.listdir(os.path.join(config.datasrc, "rgb_undistort"))
        #self.total_files = len(files)
        return context.window_manager.invoke_props_dialog(self, width = 400)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        config_id, config = _get_configuration(context.object)
        #layout.label(text="Set Sample Rate for RGB to reconstruct:")
        #box = layout.box() 
        #row = box.row()
        #row.prop(config, "sample_rate") 
        #row = box.row()
        #row.label(text="You would load around {0} cameras".format(int(self.total_files * config.sample_rate)))


class ImportReconResult(Operator):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "object_property.importreconresult"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Import Reconstruction Result"

    def execute(self, context): 
        scene = context.scene
        config_id, config = _get_configuration(context.object)
        datasrc = config.datasrc
        filepath = config.reconstructionsrc
        scene.loadreconparas.pointcloud_scale = 1.0#config.reconstructionscale
        # config.cameradisplayscale = scene.loadreconparas.camera_display_scale

        if not scene.loadreconparas.AUTOALIGN:
            load_pc(os.path.join(config.reconstructionsrc, "depthfused.ply"), scene.loadreconparas.pointcloud_scale, config_id, "reconstruction_depthfusion")
            load_reconstruction_result(filepath = filepath, 
                                        pointcloudscale = scene.loadreconparas.pointcloud_scale, 
                                        datasrc = datasrc,
                                        config_id = config_id,
                                        camera_display_scale = config.cameradisplayscale,
                                        IMPORT_RATIO = scene.loadreconparas.Import_ratio,
                                        CAMPOSE_INVERSE = config.inverse_pose
                                        )
        else:
            log_report(
            "INFO", "Starting aligning the point cloud", None
            )     

            scale =  _align_reconstruction(config, scene, scene.scalealign.THRESHOLD, scene.scalealign.NUM_THRESHOLD)
            config.reconstructionscale = 1.0#scale
            load_pc(os.path.join(config.reconstructionsrc, "depthfused.ply"), scale, config_id, "reconstruction_depthfusion")
            load_reconstruction_result(filepath = filepath, 
                                        pointcloudscale = scale, 
                                        datasrc = datasrc,
                                        config_id = config_id,
                                        camera_display_scale = config.cameradisplayscale,
                                        IMPORT_RATIO = scene.loadreconparas.Import_ratio,
                                        CAMPOSE_INVERSE = config.inverse_pose
                                        )
            

        return {'FINISHED'}
        
    def invoke(self, context, event):
    
        return context.window_manager.invoke_props_dialog(self, width = 400)
        
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        _, config = _get_configuration(context.object)
        layout.label(text="Load Image & Pose data")
        
        box = layout.box() 
        row = box.row()
        row.prop(scene.loadreconparas, "Import_ratio") 




class LoadRecon(bpy.types.PropertyGroup):
    # The properties for this class which is referenced as an 'entry' below.
    pointcloud_scale: bpy.props.FloatProperty(name="Point Cloud Display Scale", 
                                            description="Scale for display reconstruction point cloud", 
                                            default=1.00, 
                                            min=0.00, 
                                            max=1.00, 
                                            step=2, 
                                            precision=2)  
    AUTOALIGN: bpy.props.BoolProperty(name="Auto Align Point Cloud Scale", 
                                      description="Algin the Point Clound from Depth Information", 
                                      default=False)      

    Import_ratio: bpy.props.FloatProperty(name="Import Ratio", 
                                          description="Ratio to import images from campose.txt", 
                                          default=0.1, 
                                          min=0.00, 
                                          max=1.00, 
                                          step=2, 
                                          precision=2)   
                                                     
  


class WorkspaceRename(Operator):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "object_property.workspacerename"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Rename"

    def execute(self, context): 
        updateprojectname()
        return {'FINISHED'}

class RemoveWorkspace(Operator):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "object_property.removeworkspace"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Remove Current Workspace"

    def execute(self, context):
        assert context.object['type'] == 'setting'
        name  = _get_workspace_name(context.object)
        removeworkspace(name)
        for area in registeration.register.area_image_pair:
            bpy.types.SpaceView3D.draw_handler_remove(registeration.register.area_image_pair[area]["handler"], 'WINDOW')
        registeration.register.area_image_pair = {}
        return {'FINISHED'}  

class ModelICP(Operator):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "object_property.modelicp"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "Model Alignment"

    def execute(self, context):
        obj = context.object
        assert obj['type'] == 'model'
        name  = _get_workspace_name(obj)
        model_vertices = np.array([list(obj.matrix_world @ v.co) for v in obj.data.vertices])
        if name + ":reconstruction" not in bpy.data.objects:
            log_report(
                "Error", "You should upload the reconstruction result first", None
            )     
            return {'FINISHED'}
        else:
            recon = _get_reconstruction_insameworkspace(obj)
            print(recon.name)
            recon_vertices = np.array(recon["particle_coords"])
            rot = np.array(recon.matrix_world)
            recon_vertices_rotated = (rot[:3, :3].dot(recon_vertices.T) + rot[:3, [3]]).T
            trans_obj_icp = modelICP(recon_vertices_rotated, model_vertices)
            _apply_trans2obj(obj, trans_obj_icp)
        return {'FINISHED'} 

class AllModelsICP(Operator):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "object_property.allmodelsicp"  # important since its how bpy.ops.import_test.some_data is constructed
    bl_label = "All Models Alignment"

    def execute(self, context):
        recon = context.object
        assert recon['type'] == 'reconstruction'
        objects = _get_obj_insameworkspace(recon, 'model')

        for obj in objects:
            model_vertices = np.array([list(obj.matrix_world @ v.co) for v in obj.data.vertices])
            recon_vertices = np.array(recon["particle_coords"])
            rot = np.array(recon.matrix_world)
            recon_vertices_rotated = (rot[:3, :3].dot(recon_vertices.T) + rot[:3, [3]]).T
            trans_obj_icp = modelICP(recon_vertices_rotated, model_vertices)
            _apply_trans2obj(obj, trans_obj_icp)
        return {'FINISHED'}    

class CurrentDepthOperator(bpy.types.Operator):
    """Move an object with the mouse, example"""
    bl_idname = "object_property.current_depth_operator"
    bl_label = "Current depth operator"

    current_depth: FloatProperty()

    def modal(self, context, event):
        if _is_progresslabeller_object(context.object) and context.object["type"] == "camera" and event.type == 'MOUSEMOVE':
            config_id, config = _get_configuration(context.object)
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.spaces.active.region_3d.view_perspective = 'CAMERA'
                    bpy.context.scene.render.resolution_x = config.resX
                    bpy.context.scene.render.resolution_y = config.resY
                    bpy.context.scene.camera = context.object
                    cam = bpy.context.scene.camera
                    frame = cam.data.view_frame(scene = bpy.context.scene)
                    frame = [cam.matrix_world @ corner for corner in frame]
                    region = bpy.context.region
                    rv3d = bpy.context.region_data
                    frame_px = [location_3d_to_region_2d(region, rv3d, corner) for corner in frame]           
                    bias_X = min([v[0] for v in frame_px])
                    bias_Y = min([v[1] for v in frame_px])
                    res_X = max([v[0] for v in frame_px]) - min([v[0] for v in frame_px])
                    res_Y = max([v[1] for v in frame_px]) - min([v[1] for v in frame_px])
                    print(res_X, res_Y)
            if event.mouse_x > bias_X \
                and event.mouse_x < bias_X + res_X\
                and event.mouse_y > bias_Y\
                and event.mouse_x < bias_Y + res_Y:
                print(event.mouse_x, event.mouse_y) 
        return {'PASS_THROUGH'}
        

    def invoke(self, context, event):
        self.current_depth = 0
        context.window_manager.modal_handler_add(self)
        return {'FINISHED'}

class Lockcurrent3DArea(bpy.types.Operator):
    """Move an object with the mouse, example"""
    bl_idname = "object_property.lockcurrent3darea"
    bl_label = "Lock the view (progresslabelelr)"

    def execute(self, context):
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}  
    
    def modal(self, context, event):
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area_bottom = area.y
                area_left = area.x
                area_top= area.y + area.height
                area_right = area.x + area.width  
                if event.mouse_x >= area_left and event.mouse_x < area_right\
                    and event.mouse_y >= area_bottom and event.mouse_y < area_top:
                    if context.object is not None and _is_progresslabeller_object(context.object) and context.object["type"] == "camera":
                        #context.object["depth"]["UPDATEALPHA"] = True
                        context.object["rgb"]["UPDATEALPHA"] = True
                        floatscreen_handler = area.spaces[0].draw_handler_add(draw_for_area, (area, context.object), 'WINDOW', 'POST_PIXEL')
                        registeration.register.area_image_pair[area] = {"camera" : context.object,
                                                                        "handler" : floatscreen_handler}
                        print(context.object.keys())
                        log_report(
                            "Info", "Lock current 3D view area", None
                        )
                        return {'FINISHED'}
                    else:
                        log_report(
                            "Info", "Doesn't Lock anything, please select a Progresslabeller camera object", None
                        )
                        return {'FINISHED'}
        log_report(
            "Info", "Doesn't Lock anything, please select a 3D view area", None
        )
        return {'FINISHED'}

class Unlockcurrent3DArea(bpy.types.Operator):
    """Move an object with the mouse, example"""
    bl_idname = "object_property.unlockcurrent3darea"
    bl_label = "Unlock the view (progresslabelelr)"

    def execute(self, context):
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}  
    
    def modal(self, context, event):
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area_bottom = area.y
                area_left = area.x
                area_top= area.y + area.height
                area_right = area.x + area.width  
                if event.mouse_x >= area_left and event.mouse_x < area_right\
                    and event.mouse_y >= area_bottom and event.mouse_y < area_top:
                    if area in registeration.register.area_image_pair:
                        # area.spaces[0].draw_handler_remove(registeration.register.area_image_pair[area]["handler"], 'WINDOW')
                        bpy.types.SpaceView3D.draw_handler_remove(registeration.register.area_image_pair[area]["handler"], 'WINDOW')
                        registeration.register.area_image_pair.pop(area)
                        log_report(
                            "Info", "Unlock current 3D view area", None
                        )
                        return {'FINISHED'}
                    else:
                        log_report(
                            "Info", "Current 3D view area is not lock", None
                        )
                        return {'FINISHED'}                        
        log_report(
            "Info", "Doesn't Unlock anything, please select a 3D view area", None
        )
        return {'FINISHED'}

            

def register():
    bpy.utils.register_class(PlaneAlignment)
    bpy.utils.register_class(PlaneAlignmentConfig)
    bpy.utils.register_class(MaskOptimizationConfig)
    bpy.types.Object.optimizationparas = bpy.props.PointerProperty(type=MaskOptimizationConfig) 
    bpy.types.Scene.planalignmentparas = bpy.props.PointerProperty(type=PlaneAlignmentConfig) 
    bpy.utils.register_class(ImportCamRGBDepth)
    bpy.utils.register_class(ImportReconResult)
    bpy.utils.register_class(LoadRecon)
    bpy.types.Scene.loadreconparas = bpy.props.PointerProperty(type=LoadRecon)  
    
    bpy.utils.register_class(ModelICP)
    bpy.utils.register_class(AllModelsICP)

    bpy.utils.register_class(WorkspaceRename)
    bpy.utils.register_class(RemoveWorkspace)

    bpy.utils.register_class(CurrentDepthOperator)
    bpy.utils.register_class(Lockcurrent3DArea)
    bpy.utils.register_class(Unlockcurrent3DArea)
    
    bpy.utils.register_class(MaskOptimization)
    

def unregister():
    bpy.utils.unregister_class(PlaneAlignment)
    bpy.utils.unregister_class(PlaneAlignmentConfig)
    bpy.utils.unregister_class(MaskOptimizationConfig)
    bpy.utils.unregister_class(ImportCamRGBDepth)
    bpy.utils.unregister_class(ImportReconResult)
    bpy.utils.unregister_class(LoadRecon)
    bpy.utils.unregister_class(ModelICP)
    bpy.utils.unregister_class(AllModelsICP)
    
    bpy.utils.unregister_class(WorkspaceRename)
    bpy.utils.unregister_class(RemoveWorkspace)
    bpy.utils.unregister_class(CurrentDepthOperator)
    bpy.utils.unregister_class(Lockcurrent3DArea)
    bpy.utils.unregister_class(Unlockcurrent3DArea)
    
    bpy.utils.unregister_class(MaskOptimization)
    
    

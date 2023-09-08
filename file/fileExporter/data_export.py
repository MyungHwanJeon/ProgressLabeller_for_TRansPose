import bpy
import json
import os
import numpy as np
import bpy
import json
import os
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import trimesh
import pyrender
import multiprocessing
import subprocess
import sys

from kernel.exporter import data_export
from kernel.logging_utility import log_report

# ImportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, EnumProperty
from bpy.types import Operator
from kernel.exporter import configuration_export, objectposes_export
from kernel.blender_utility import \
    _get_configuration, _get_reconstruction_insameworkspace, _get_obj_insameworkspace, _get_workspace_name, _apply_trans2obj, \
    _align_reconstruction, _is_progresslabeller_object
import registeration.register 

class PoseOptimization(Operator, ExportHelper):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "export_data.poseoptimization" 
    bl_label = "Pose Optimization"


    def execute(self, context):
    
        setting_object = 0
        visible_objects = [obj for obj in bpy.context.view_layer.objects if obj.visible_get()]
        for obj in visible_objects:
            if obj.name == "untitled:Setting":
                setting_object = obj


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
        code_path = os.path.join(source, "../", "offline", "mask_opt.py")
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



class MaskOutput(Operator, ExportHelper):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "export_data.savemask" 
    bl_label = "Save Mask"


    def execute(self, context):

        setting_object = 0
        visible_objects = [obj for obj in bpy.context.view_layer.objects if obj.visible_get()]
        for obj in visible_objects:
            if obj.name == "untitled:Setting":
                setting_object = obj


        config_id = setting_object['config_id']
        config = bpy.context.scene.configuration[config_id]
        # config.keys()
        # ['projectname', 'modelsrc', 'reconstructionsrc', 'datasrc', 'resX', 'resY', 'fx', 'fy', 'cx', 'cy', 'lens', 'inverse_pose', 'reconstructionscale', 'cameradisplayscale', 'recon_trans', 'sample_rate', 'depth_scale', 'depth_ignore']


        path = setting_object['dir']
        output_filepath = (setting_object["dir"] if setting_object["dir"].endswith("/") else setting_object["dir"] + "/") + "output"
        cam_pose = np.zeros(7, dtype=np.float32)
        obj_pose = np.zeros(7, dtype=np.float32)
        obj_context = 0
        cam_context = 0

        if context.selected_objects[0]["type"] == "camera" and context.selected_objects[1]["type"] == "model":

            cam_context = context.selected_objects[0]
            obj_context = context.selected_objects[1]
            	
            cam_pose[0] = context.selected_objects[0].location[0]
            cam_pose[1] = context.selected_objects[0].location[1]
            cam_pose[2] = context.selected_objects[0].location[2]

            cam_pose[3] = context.selected_objects[0].rotation_quaternion[0]
            cam_pose[4] = context.selected_objects[0].rotation_quaternion[1]
            cam_pose[5] = context.selected_objects[0].rotation_quaternion[2]
            cam_pose[6] = context.selected_objects[0].rotation_quaternion[3]

            obj_pose[0] = context.selected_objects[1].location[0]
            obj_pose[1] = context.selected_objects[1].location[1]
            obj_pose[2] = context.selected_objects[1].location[2]
            
            obj_context.rotation_mode = 'QUATERNION'

            obj_pose[3] = context.selected_objects[1].rotation_quaternion[0]
            obj_pose[4] = context.selected_objects[1].rotation_quaternion[1]
            obj_pose[5] = context.selected_objects[1].rotation_quaternion[2]
            obj_pose[6] = context.selected_objects[1].rotation_quaternion[3]



        elif context.selected_objects[1]["type"] == "camera" and context.selected_objects[0]["type"] == "model":

            cam_context = context.selected_objects[1]
            obj_context = context.selected_objects[0]

            cam_pose[0] = context.selected_objects[1].location[0]
            cam_pose[1] = context.selected_objects[1].location[1]
            cam_pose[2] = context.selected_objects[1].location[2]

            cam_pose[3] = context.selected_objects[1].rotation_quaternion[0]
            cam_pose[4] = context.selected_objects[1].rotation_quaternion[1]
            cam_pose[5] = context.selected_objects[1].rotation_quaternion[2]
            cam_pose[6] = context.selected_objects[1].rotation_quaternion[3]

            obj_pose[0] = context.selected_objects[0].location[0]
            obj_pose[1] = context.selected_objects[0].location[1]
            obj_pose[2] = context.selected_objects[0].location[2]
            
            obj_context.rotation_mode = 'QUATERNION'

            obj_pose[3] = context.selected_objects[0].rotation_quaternion[0]
            obj_pose[4] = context.selected_objects[0].rotation_quaternion[1]
            obj_pose[5] = context.selected_objects[0].rotation_quaternion[2]
            obj_pose[6] = context.selected_objects[0].rotation_quaternion[3]

            #obj_context = context.selected_objects[0]
            #cam_context = context.selected_objects[1]

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
                "name":cam_context.name.split(":")[1],
                "resolution": [config.resX, config.resY],
                "intrinsic": [[config.fx, 0, config.cx],
                              [0, config.fy, config.cy],
                              [0, 0, 1]],
                "inverse_pose": False,
                "lens": 30.0, 
                "translation":[float(cam_pose[0]), float(cam_pose[1]), float(cam_pose[2])],
                "quaternion":[float(cam_pose[3]), float(cam_pose[4]), float(cam_pose[5]), float(cam_pose[6])],
            },
            'object':{
                "name":obj_context.name.split(":")[1],
                "path":obj_context["path"],
                "translation":[float(obj_pose[0]), float(obj_pose[1]), float(obj_pose[2])],
                "quaternion":[float(obj_pose[3]), float(obj_pose[4]), float(obj_pose[5]), float(obj_pose[6])],
            },
            'reconstruction':{
	        "scale": 1.0,
                "cameradisplayscale": 0.01,
                "recon_trans": "1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,1;"
            },
            'data':{
                "sample_rate": 0.1,
                "depth_scale": 0.001,
                "depth_ignore": 8.0
            }
        }


        with open("/tmp/progresslabeller_mask.json", "w") as f:
            json.dump(output_dict, f, indent = True)

        source = os.path.dirname(os.path.dirname(__file__))
        code_path = os.path.join(source, "../", "offline", "main_mask.py")
        subprocess.call("{} {} {} {} {}".format(sys.executable, code_path, "/tmp/progresslabeller_mask.json", output_filepath, "Yourtype"), shell=True)

        os.remove("/tmp/progresslabeller_mask.json")

        log_report("Info", "Save Mask", None)  
        return {'FINISHED'}


   



class DataOutput(Operator, ExportHelper):
    """This appears in the tooltip of the operator and in the generated docs"""
    bl_idname = "export_data.dataoutput" 
    bl_label = "Output Data"

    filename_ext = "/"

    filter_glob: StringProperty(
        default="/",
        options={'HIDDEN'},
        maxlen=255,  
    )

    dataformatType: EnumProperty(
        name="Output format",
        description="Choose a output format",
        items=(
            ('BOP', "BOP", "BOP challenge format"),
            ('YCBV', "YCBV", "YCBV dataset format"),
            ('ProgressLabeller', "ProgressLabeller", "ProgressLabeller format"),
            ('Yourtype', "Yourtype", "Your own form (please define your own type first)"),
        ),
        default='ProgressLabeller',
    )

    def execute(self, context):

        assert context.object['type'] == 'setting'
        config_id = context.object['config_id']
        path = context.object['dir']
        config = bpy.context.scene.configuration[config_id]

        files = os.listdir(config.reconstructionsrc)
        if "campose_R.txt" not in files:
            log_report(
                "Error", "Please do reconstruction first", None
            )
        elif "label_pose.yaml" not in files:
            log_report(
                "Error", "Please allocate the object in the scene and save object poses", None
            )
        if not os.path.exists(os.path.join(config.modelsrc, "object_label.json")):
            log_report(
                "Error", "Please define your models' label fine in <PATH/TO/MODEL>/object_label.json", None
            )            
        else:
            configuration_export(config, "/tmp/progresslabeller.json")
            log_report(
                "Info", "Export data to" + self.filepath, None
            )
            #print(self.dataformatType)
            data_export("/tmp/progresslabeller.json", self.filepath, self.dataformatType)
            
            os.remove("/tmp/progresslabeller.json")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        if "dir" in context.object:
            self.filepath = (context.object["dir"] if context.object["dir"].endswith("/") else context.object["dir"] + "/") + "output"
        # Tells Blender to hang on for the slow user input
        return {'RUNNING_MODAL'}



def register():
    bpy.utils.register_class(PoseOptimization)
    bpy.utils.register_class(MaskOutput)
    bpy.utils.register_class(DataOutput)

def unregister():
    bpy.utils.register_class(PoseOptimization)
    bpy.utils.unregister_class(MaskOutput)
    bpy.utils.unregister_class(DataOutput)

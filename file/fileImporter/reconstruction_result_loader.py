import bpy
from kernel.loader import load_reconstruction_result

# ImportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, EnumProperty, FloatProperty
from bpy.types import Operator


class ImportReconstruction(Operator, ImportHelper):
    """Load model for pose alignment and segmentation"""
    bl_idname = "import_data.reconstruction"  
    bl_label = "Import Reconstruction Result"

    filename_ext = "/"

    filter_glob: StringProperty(
        default="/",
        options={'HIDDEN'},
        maxlen=255,  
    )

    reconstruction_method: EnumProperty(
        name="Reconstruction Method",
        description="Choose the method you use for your reconstruction data",
        items=(
            ('COLMAP', "COLMAP", "COLMAP"),
            ('KinectFusion', "KinectFusion", "KinectFusion"),
            ('Meshroom', "Meshroom", "Meshroom"),
        ),
        default='COLMAP',
    )    

    pointcloudscale: FloatProperty(
        name="PC Scale", 
        description="scale for the loading point cloud", 
        default=0.2, min=0.00, 
        max=1.00, step=2, precision=2)

    camerascale: FloatProperty(
        name="Camera Scale", 
        description="scale for displaying the camera", 
        default=0.1, min=0.00, 
        max=1.00, step=2, precision=2)

    def execute(self, context):
        load_reconstruction_result(filepath = self.filepath, 
                                   reconstruction_method = self.reconstruction_method,
                                   pointcloudscale = self.pointcloudscale,
                                   imagesrc = bpy.context.scene.configuration.imagesrc,
                                   camera_display_scale = self.camerascale,)
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        if hasattr(bpy.context.scene.configuration, 'reconstructionsrc'):
            self.filepath = bpy.context.scene.configuration.reconstructionsrc
        return {'RUNNING_MODAL'}

def _menu_func_import(self, context):
    self.layout.operator(ImportReconstruction.bl_idname, text="ProgressLabeller Load Reconstruction Result(package)")



def register():
    bpy.utils.register_class(ImportReconstruction)


def unregister():
    bpy.utils.unregister_class(ImportReconstruction)
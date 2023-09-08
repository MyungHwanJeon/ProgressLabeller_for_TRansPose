import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse import offlineParam
#from render import offlineRender
from render_custom import offlineRender
from offlineRecon import offlineRecon

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
    data_format = sys.argv[3]
    param = offlineParam(config_path)
    interpolation_type = "all"
    #offlineRecon(param, interpolation_type)
    #offlineRender(param, output_dir, interpolation_type, pkg_type=data_format)  
    offlineRender(param, output_dir)  
    

from .description import *
from .diff import DiffParts
from .object_moving import RepMovingModel
from .object_resizing import RepResizingModel
from .object_removing import RepReMovingModel
from .content_dragging import ContentDragModel
from .score import RepScoreModel
from .forensics_accessibility import RepForensicsAccessModel, RepSaveForensicsAccessModel
from .detection_res import DetectionRes, SaveDetectionRes

# # 读取当前文件夹下的所有py文件, 自动import所有基类为from pydantic import BaseModel的类

# import os
# import importlib.util
# import sys
# from pydantic import BaseModel


# def import_pydantic_models_from_directory(directory="."):
#     imported_classes = set()  # Set to keep track of imported class names

#     # List all files in the given directory
#     for filename in os.listdir(directory):
#         # Check if the file is a Python file
#         if filename.endswith(".py") and filename != os.path.basename(__file__):
#             module_name = filename[:-3]  # Remove the .py extension
#             file_path = os.path.join(directory, filename)

#             # Load the module
#             spec = importlib.util.spec_from_file_location(module_name, file_path)
#             module = importlib.util.module_from_spec(spec)
#             sys.modules[module_name] = module
#             spec.loader.exec_module(module)

#             # Import classes that inherit from BaseModel
#             for attr_name in dir(module):
#                 attr = getattr(module, attr_name)
#                 if isinstance(attr, type) and issubclass(attr, BaseModel) and attr is not BaseModel and attr_name not in imported_classes:
#                     globals()[attr_name] = attr
#                     imported_classes.add(attr_name)  # Add to the set of imported classes
#                     print(f"Imported {attr_name} from {filename}")


# # Call the function to import models
# import_pydantic_models_from_directory()

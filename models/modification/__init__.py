# 找到当前文件夹下的所有继承于from pydantic import BaseModel的子类, 并将其import出来
import os
import sys
import importlib
from typing import List
from pydantic import BaseModel


def get_subclasses(base_class):
    subclasses = []
    for module in sys.modules.values():
        for name, obj in module.__dict__.items():
            if isinstance(obj, type) and issubclass(obj, base_class) and obj != base_class:
                subclasses.append(obj)
    return subclasses


def import_subclasses(base_class):
    subclasses = get_subclasses(base_class)
    for subclass in subclasses:
        importlib.import_module(subclass.__module__)


import_subclasses(BaseModel)

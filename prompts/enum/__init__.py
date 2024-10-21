from typing import Dict, List, ClassVar
from pydantic import BaseModel, Field
from enum import Enum
from ..model.object_info import ObjectInfo
from typing import Optional


class ModifyType(Enum):
    REMOVE = "remove_object"
    COPY = "copy_object"
    RECOLORED_MODIFY = "recolored_object"
    OBJECT_RESIZING = "resizing_object"
    OBJECT_MOVING = "moving_object"
    OBJECT_PASTING = "pasting_object"

    @classmethod
    def choices(cls):
        info = [member.value for member in cls]
        return ",".join(info)

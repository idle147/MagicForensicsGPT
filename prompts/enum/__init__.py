from typing import Dict, List, ClassVar
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class ModifyType(Enum):
    OBJECT_RESIZING = "resizing_object"
    OBJECT_MOVING = "moving_object"
    OBJECT_REMOVING = "removing_object"
    CONTENT_DRAGGING = "content_dragging"

    @classmethod
    def choices(cls):
        info = [member.value for member in cls]
        return ",".join(info)

    @classmethod
    def get_modify(cls):
        info = [member for member in cls]
        return info

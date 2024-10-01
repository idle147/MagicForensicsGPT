from typing import Dict, List
from pydantic import BaseModel, Field
from enum import Enum
from .object_info import ObjectInfo


class ModifyType(Enum):
    REMOVE = "remove object"
    COPY = "copy object"
    MOVE = "move object"
    RECOLORED_MODIFY = "recolored object"
    RESIZE_MODIFY = "resize object"

    @classmethod
    def choices(cls):
        info = [member.value for member in cls]
        return ",".join(info)


class ModifyModel(BaseModel):
    mode: str = Field(description="Image Editing mode")
    procedure: str = Field(description="Image Editing Procedure")
    # captions: List[str] = Field(description="Five items are needed for the revised description(captions) of the entire image.")


class EditInfoModel(BaseModel):
    object_info: ObjectInfo = Field(description="Object information")
    modify: Dict[str, str] = Field(description="Modify information dictionary with mode as key and procedure as value")

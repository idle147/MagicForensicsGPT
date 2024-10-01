from typing import Dict, List, ClassVar
from pydantic import BaseModel, Field
from enum import Enum
from .object_info import ObjectInfo
from typing import Optional


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


# class ModifyModel(BaseModel):
#     mode: str = Field(description="Image Editing mode")
#     procedure: str = Field(description="Image Editing Procedure")
#     # captions: List[str] = Field(description="Five items are needed for the revised description(captions) of the entire image.")


# class EditInfoModel(BaseModel):
#     object_info: ObjectInfo = Field(description="Object information")
#     modify: Dict[str, str] = Field(description="Modify information dictionary with mode as key and procedure as value")


class RepMoveModel(BaseModel):
    object_mask_path: str = Field(description="Path of the object mask")
    object_text_info: ObjectInfo = Field(description="Object information in text")
    editing_procedure: str = Field(description="Image Editing Procedure")
    start_point: List[int] = Field(description="Start point of the object")
    end_point: List[int] = Field(description="End point of the object")
    end_mask_pos: List[int] = Field(description="End mask pos of the object")


class SaveMoveModel(RepMoveModel):
    modify_type: ClassVar[ModifyType] = ModifyType.MOVE
    image_path: str = Field(description="Path of the image")
    mask_path: str = Field(description="Path of the mask")
    reference_mask_path: Optional[str] = Field(description="Path of the reference image")

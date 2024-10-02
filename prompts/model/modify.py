from typing import Dict, List, ClassVar
from pydantic import BaseModel, Field
from enum import Enum
from .object_info import ObjectInfo
from typing import Optional


class ModifyType(Enum):
    REMOVE = "remove object"
    COPY = "copy object"
    RECOLORED_MODIFY = "recolored object"
    OBJECT_RESIZING = "resizing object"
    OBJECT_MOVING = "moving object"
    OBJECT_PASTING = "pasting new object"

    @classmethod
    def choices(cls):
        info = [member.value for member in cls]
        return ",".join(info)


class RepResizingModel(BaseModel):
    editing_procedure: str = Field(description="Resizing Procedure")


class RepMovingModel(BaseModel):
    editing_procedure: str = Field(description="Moving Procedure")
    end_point: List[float] = Field(
        description="The end position of the center point of the moving object. Be careful not to go beyond the picture boundaries"
    )

    def need_scales(self):
        return ["end_point"]


class RepPastingModel(BaseModel):
    editing_procedure: str = Field(description="Image Editing Procedure.")
    pasting_point: List[float] = Field(description="A central position where you need to paste it.")

    def need_scales(self):
        return ["pasting_point"]


class SaveMoveModel(RepMovingModel):
    modify_type: ClassVar[ModifyType] = ModifyType.OBJECT_MOVING
    image_path: str = Field(description="Path of the image")
    mask_path: str = Field(description="Path of the mask")
    reference_mask_path: Optional[str] = Field(description="Path of the reference image")

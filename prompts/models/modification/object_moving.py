from typing import List, ClassVar
from pydantic import BaseModel, Field
from typing import Optional
from ..enum import ModifyType


class RepMovingModel(BaseModel):
    editing_procedure: str = Field(description="Object Move Procedure Description")
    edited_results: str = Field(description="Description of the edited image effects")
    end_point: List[float] = Field(
        description="The end position of the center point of the moving object. Be careful not to go beyond the picture boundaries"
    )

    def need_scales(self):
        return ["end_point"]


class SaveMoveModel(RepMovingModel):
    modify_type: ClassVar[ModifyType] = ModifyType.OBJECT_MOVING
    image_path: str = Field(description="Path of the image")
    mask_path: str = Field(description="Path of the mask")
    reference_mask_path: Optional[str] = Field(description="Path of the reference image")

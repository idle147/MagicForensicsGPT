from typing import Dict, List, ClassVar
from pydantic import BaseModel, Field
from enum import Enum
from .object_info import ObjectInfo
from typing import Optional


class RepResizingModel(BaseModel):
    editing_procedure: str = Field(description="Resizing Object Procedure Description")
    edited_results: str = Field(description="Description of the edited image effects")
    resizing_scale: float = Field(description="Resizing Scaling")

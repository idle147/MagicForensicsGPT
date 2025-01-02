from typing import Dict, List, ClassVar
from pydantic import BaseModel, Field
from enum import Enum
from ..description.object_information import ObjectInformation
from typing import Optional


class RepRecoloringModel(BaseModel):
    editing_procedure: str = Field(description="Resizing Object Procedure Description")
    edited_results: str = Field(description="Description of the edited image effects")
    recoloring_result: str = Field(description="Recoloring Procedure Description")

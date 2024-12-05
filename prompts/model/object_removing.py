from pydantic import BaseModel, Field
from typing import List, Optional


class RepReMovingModel(BaseModel):
    editing_procedure: str = Field(description="Object Removal Procedure Description")
    edited_results: str = Field(description="Description of the edited image effects")
    end_point: Optional[List[float]] = Field(
        description="The end position of the center point of the moving object. Be careful not to go beyond the picture boundaries"
    )

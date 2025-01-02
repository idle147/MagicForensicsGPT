from typing import List
from pydantic import BaseModel, Field


class RepGenerationModel(BaseModel):
    editing_procedure: str = Field(
        description="A detailed description of the object generation procedure. e.g 'Add a large brown dog sitting on the grass'."
    )
    edited_results: str = Field(
        description="A comprehensive description of the visual effects and changes in the image after editing. Include details about the object's appearance, lighting, and integration with the environment."
    )
    end_point: List[float] = Field(
        description="Coordinates representing the final position of the center point of the generated object within the image. Ensure these coordinates are within the image boundaries."
    )

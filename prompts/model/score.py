from pydantic import BaseModel, Field


class RegenerationModel(BaseModel):
    need_regeneration: bool = Field(description="Is the edited image need to be regenerated?")
    method: str = Field(description="How to regenerate the edited image?")


class ScoreModel(BaseModel):
    score: float = Field(description="The edited image is scored on a scale from 0 to 10.")
    detail_analysis: str = Field(description="What is the reason for the score given to the edited image?")
    # regeneration: RegenerationModel = Field(description="Whether the image needs to be regenerated?")
    # editing_guidance: str = Field(description="How to paste the edited content back into the original image using the mask?")

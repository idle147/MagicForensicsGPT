from pydantic import BaseModel, Field


class ScoreModel(BaseModel):
    score: float = Field(description="The edited image is scored on a scale from 0 to 10.")
    reason: str = Field(description="What is the reason for the score given to the edited image?")
    defect: str = Field(description="What defects are present in the edited image?")

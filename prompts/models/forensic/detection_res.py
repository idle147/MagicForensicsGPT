from pydantic import BaseModel, Field


class DetectionRes(BaseModel):
    real_or_fake: str = Field(default="Answer real or fake. The answer to whether the image has been tampered with.", required=True)
    reason_for_thinking: str = Field(default="The reason for thinking this way.", required=True)


class SaveDetectionRes(DetectionRes):
    image_path: str = Field(default="")

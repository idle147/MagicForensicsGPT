from pydantic import BaseModel, Field
from enum import Enum


# 定义一个枚举类
class DifficultyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RepForensicsAccessModel(BaseModel):
    full_image_desc: str = Field(description="A detailed description of the image")
    edited_area: str = Field(description="A detailed description of the edited areas if image is edited else blank")
    high_feature: str | dict = Field(description="Results of high-level feature analysis")
    mid_feature: str | dict = Field(description="Results of mid-level feature analysis")
    low_feature: str | dict = Field(description="Results of low-level feature analysis")
    level: DifficultyLevel | dict = Field(description="The difficulty level of performing digital forensics on the edited image")
    conclusion: str | dict = Field(description="Conclusion and reasons for determining the difficulty level")


class RepSaveForensicsAccessModel(RepForensicsAccessModel):
    real_or_fake: str = Field(default="")
    image_path: str = Field(default="")
    mask_path: str = Field(default="")

from pydantic import BaseModel, Field
from enum import Enum


# 定义一个枚举类
class DifficultyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RepForensicsAccessModel(BaseModel):
    full_image_desc: str = Field(description="A detailed description of the entire image")
    category_desc: str = Field(description="Image category only")
    other_desc: str = Field(description="Description of other regulations")
    edited_area: str = Field(description="A detailed description of the edited areas if image is edited else blank")
    high_feature: str = Field(description="Results of high-level feature analysis")
    mid_feature: str = Field(description="Results of mid-level feature analysis")
    low_feature: str = Field(description="Results of low-level feature analysis")
    level: DifficultyLevel = Field(description="The difficulty level of performing digital forensics on the edited image")
    conclusion: str = Field(description="Conclusion and reasons for determining the difficulty level")


class RepSaveForensicsAccessModel(BaseModel):
    real_or_fake: str = Field(default="", title="Real or Fake", description="Indicates whether the forensics access model is real or fake.")
    image_path: str = Field(default="", title="Image Path", description="Path to the image used for forensics.")
    mask_path: str = Field(default="", title="Mask Path", description="Path to the mask used for the image.")
    with_ref: dict = Field(default={}, title="With Reference", description="Forensics access model with reference data.")
    without_ref: dict = Field(
        default={}, title="Without Reference", description="Forensics access model without reference data."
    )

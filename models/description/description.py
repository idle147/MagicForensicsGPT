from typing import List
from .object_information import ObjectInformation
from pydantic import BaseModel, Field


class DescriptionModel(BaseModel):
    caption: str = Field(description="Title of the natural picture")
    brief: str = Field(description="A brief description of the picture")
    detail: str = Field(description="A detail description of the picture")
    object_infos: List[ObjectInformation] = Field(description="Every objects in the picture")
    mask_info: ObjectInformation = Field(description="objects in the mask, Don't give coordinates")

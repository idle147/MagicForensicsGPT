from typing import List
from .object_info import ObjectInfo
from pydantic import BaseModel, Field


class DescriptionModel(BaseModel):
    title: str = Field(description="Title of the natural picture")
    brief: str = Field(description="A brief description of the picture")
    detail: str = Field(description="A detail description of the picture")
    object_info: List[ObjectInfo] = Field(description="Every objects in the picture")
    mask_object_info: List[ObjectInfo] = Field(description="objects in the mask")


class SaveDescriptionModel(DescriptionModel):
    captions: List[str] = Field(description="Captions of the picture")

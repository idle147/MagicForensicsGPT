from typing import List
from .object_info import ObjectInfo
from pydantic import BaseModel, Field


class DescriptionModel(BaseModel):
    title: str = Field(description="Title of the natural picture")
    brief: str = Field(description="A brief description of the picture")
    detail: str = Field(description="A detail description of the picture")
    object_info: List[ObjectInfo] = Field(description="Every objects in the picture")
    captions: List[str] = Field(description="Captions of the picture. This item is user input, and the output is an empty")

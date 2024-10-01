from pydantic import BaseModel, Field


class ObjectInfo(BaseModel):
    object: str = Field(description="Image Editing object, e.g. 'person, car'")
    object_referring: str = Field(
        description="Object referring to the image editing object, e.g. 'The man sitting in the second chair from the left by the lake'"
    )

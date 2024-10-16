from typing import List
from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float = Field(description="The x-coordinate of the point.")
    y: float = Field(description="The y-coordinate of the point.")


class Polygon(BaseModel):
    points: List[Point] = Field(description="A list of points that form a polygon. It has to have at least 16 points.")


class DiffParts(BaseModel):
    different_part: List[Polygon] = Field(
        description=(
            "List of different parts of the image. Each part consists of several position points, "
            "and these position points can be connected to form a shape. Each part must have more "
            "than 16 position points."
        ),
    )

    def extract_polygons(self) -> List[List[tuple]]:
        """Extracts the list of polygons in the desired format."""
        return [[(point.x, point.y) for point in polygon.points] for polygon in self.different_part]

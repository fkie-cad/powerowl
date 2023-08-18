import math
from typing import Tuple, List

from shapely import Polygon


class Position:
    def __init__(self, x: float, y: float, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, position: 'Position'):
        return math.sqrt(
            (position.x - self.x) ** 2 +
            (position.y - self.y) ** 2 +
            (position.z - self.z) ** 2
        )

    def middle(self, position: 'Position'):
        return Position(
            x=(self.x + position.x) / 2,
            y=(self.y + position.y) / 2,
            z=(self.z + position.z) / 2
        )

    def center_2d(self, other_positions: List['Position']) -> 'Position':
        if len(other_positions) == 0:
            return self
        if len(other_positions) == 1:
            return self.middle(other_positions[0])
        positions = [(self.x, self.y)]
        positions.extend([(pos.x, pos.y) for pos in other_positions])
        polygon = Polygon(positions).convex_hull
        centroid = polygon.centroid.coords[0]
        return Position(centroid[0], centroid[1])

    @staticmethod
    def static_center_2d(positions: List['Position']):
        if len(positions) == 0:
            raise ValueError("Cannot calculate center of no positions")
        return positions[0].center_2d(positions[1:])

    def to_tuple_2d(self):
        return self.x, self.y

    def to_tuple_3d(self):
        return self.x, self.y, self.z

    @staticmethod
    def from_tuple(pos: Tuple):
        if len(pos) == 2:
            return Position(pos[0], pos[1])
        elif len(pos) == 3:
            return Position(pos[0], pos[1], pos[2])
        raise ValueError(f"Invalid position tuple: {pos} - Expected two or three values")

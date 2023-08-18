from abc import ABC

from .bus import Bus
from .grid_element import GridElement


class GridAsset(GridElement, ABC):
    """
    A grid element that belongs to a single grid node (i.e., bus)
    """
    prefix: str = "grid-asset"

    def get_bus(self) -> 'Bus':
        return self.get_property_value("bus")

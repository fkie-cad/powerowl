from abc import ABC

from .grid_element import GridElement


class GridNode(GridElement, ABC):
    """
    A grid element that serves as a node with outgoing edges.
    Usually a Bus.
    """
    prefix: str = "grid-node"

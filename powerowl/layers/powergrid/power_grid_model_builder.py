from typing import TYPE_CHECKING

from powerowl.exceptions.grid_element_already_exists_exception import GridElementAlreadyExistsException
from .elements import Bus, Line
from .elements.grid_element import GridElement

if TYPE_CHECKING:
    from .power_grid_model import PowerGridModel


class PowerGridModelBuilder:
    def __init__(self, model: 'PowerGridModel'):
        self._model = model

    def create_elem(self, cls, **kwargs):
        e = cls(**kwargs)
        self.add_elem(e)

    def add_elem(self, e: GridElement):
        cls_name = e.__class__.__name__
        e_type = e.prefix
        if e.index in self._model.elements.setdefault(e_type, {}):
            raise GridElementAlreadyExistsException(f"{cls_name} {e.index} already exists in Model")
        self._model.elements.setdefault(e_type, {})[e.index] = e

    def add_bus(self, bus: Bus):
        self.add_elem(bus)

    def create_bus(self, **kwargs):
        self.create_elem(Bus, **kwargs)

    def add_line(self, line: Line):
        self.add_elem(line)

    def create_line(self, **kwargs):
        self.create_elem(Line, **kwargs)

    def get_element(self, element_type, element_index) -> GridElement:
        return self._model.elements[element_type][element_index]

from typing import List, Union

import numpy as np

from . import GridElement
from .attribute_specification import AttributeSpecification as As
from .enums.switch_type import SwitchType
from .grid_annotator import GridAnnotator
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class Switch(GridAnnotator):
    prefix = "switch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False),
            As("observable", Gvc.GENERIC, bool, True, required=False),
            As("controllable", Gvc.PROPERTY, bool, True, required=False),

            As("bus", Gvc.PROPERTY, GridNode, None),
            As("element", Gvc.PROPERTY, GridElement, None),
            As("type", Gvc.PROPERTY, SwitchType, SwitchType.NONE),
            As("maximum_current", Gvc.PROPERTY, float, np.NAN, Unit.AMPERE, Scale.BASE, required=False),

            As("is_closed", Gvc.MEASUREMENT, bool, True, simulation_context=Gvc.CONFIGURATION, related=[(Gvc.CONFIGURATION, "closed")]),

            As("in_service", Gvc.CONFIGURATION, bool, True, required=False),
            As("closed", Gvc.CONFIGURATION, bool, True, related=[(Gvc.MEASUREMENT, "is_closed")])
        ]

    def et(self):
        return self.get_property("element").value.prefix

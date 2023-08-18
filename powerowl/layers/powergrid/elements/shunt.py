from typing import List

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .grid_asset import GridAsset
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class Shunt(GridAsset):
    prefix = "shunt"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("bus", Gvc.PROPERTY, GridNode, None),
            As("rated_voltage", Gvc.PROPERTY, float, np.NAN, Unit.VOLT, Scale.BASE),
            As("active_power", Gvc.PROPERTY, float, np.NAN, Unit.WATT, Scale.BASE),
            As("reactive_power", Gvc.PROPERTY, float, np.NAN, Unit.VAR, Scale.BASE),

            As("step", Gvc.CONFIGURATION, int, 1),
            As("in_service", Gvc.CONFIGURATION, bool, True),

            As("active_power", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE),
            As("reactive_power", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE),
            As("voltage", Gvc.MEASUREMENT, float, np.NAN, Unit.PER_UNIT)
        ]

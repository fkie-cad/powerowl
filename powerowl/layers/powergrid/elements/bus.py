from typing import List, Tuple

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .enums.bus_type import BusType
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class Bus(GridNode):
    prefix = "bus"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("voltage_niveau", Gvc.PROPERTY, float, 0, Unit.VOLT, Scale.BASE),
            As("position", Gvc.PROPERTY, Tuple, None, required=False, pp_column="coords"),
            As("geo_position", Gvc.PROPERTY, Tuple[float, float], None, required=False, pp_column="geodata"),
            As("type", Gvc.PROPERTY, BusType, BusType.BUSBAR, pp_column="type"),
            As("zone", Gvc.PROPERTY, str, None, required=False, pp_column="zone"),
            As("minimum_voltage", Gvc.PROPERTY, float, 0.9, Unit.PER_UNIT, required=False),
            As("maximum_voltage", Gvc.PROPERTY, float, 1.1, Unit.PER_UNIT, required=False),

            As("in_service", Gvc.CONFIGURATION, bool, True, required=True),

            As("voltage", Gvc.MEASUREMENT, float, np.NAN, Unit.PER_UNIT),
            As("voltage_angle", Gvc.MEASUREMENT, float, 0, Unit.DEGREE, Scale.BASE),
            As("active_power", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE),
            As("reactive_power", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE)
        ]

from typing import List, Tuple

import numpy as np
from powerowl.layers.powergrid.elements.enums.grid_element_role import GridElementRole

from .attribute_specification import AttributeSpecification as As
from .enums.bus_type import BusType
from .enums.voltage_niveau import VoltageNiveau
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class Bus(GridNode):
    prefix = "bus"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_buses(self) -> List['Bus']:
        return [self]

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("controllable", Gvc.PROPERTY, bool, True, required=False),
            As("voltage_niveau", Gvc.PROPERTY, float, 0, Unit.VOLT, Scale.BASE),
            As("position", Gvc.PROPERTY, Tuple, None, required=False, pp_column="coords"),
            As("geo_position", Gvc.PROPERTY, Tuple[float, float], None, required=False, pp_column="geo"),
            As("type", Gvc.PROPERTY, BusType, BusType.BUSBAR, pp_column="type"),
            As("role", Gvc.PROPERTY, GridElementRole, GridElementRole.AUTO),
            As("zone", Gvc.PROPERTY, str, None, required=False, pp_column="zone"),
            As("minimum_voltage", Gvc.PROPERTY, float, 0.9, Unit.PER_UNIT, required=False),
            As("maximum_voltage", Gvc.PROPERTY, float, 1.1, Unit.PER_UNIT, required=False),

            As("in_service", Gvc.CONFIGURATION, bool, True, required=True, operator_controllable=False),

            As("voltage", Gvc.MEASUREMENT, float, np.nan, Unit.PER_UNIT),
            As("voltage_angle", Gvc.MEASUREMENT, float, 0, Unit.DEGREE, Scale.BASE),
            As("active_power", Gvc.MEASUREMENT, float, np.nan, Unit.WATT, Scale.BASE),
            As("reactive_power", Gvc.MEASUREMENT, float, np.nan, Unit.VAR, Scale.BASE)
        ]

    def get_voltage_niveau(self) -> VoltageNiveau:
        return VoltageNiveau.by_value(self.get_property_value("voltage_niveau"))

    def is_muff(self) -> bool:
        return self.get_property_value("type") == BusType.MUFF

    def is_bus_bar(self) -> bool:
        return self.get_property_value("type") == BusType.BUSBAR

    def is_node(self) -> bool:
        return self.get_property_value("type") == BusType.NODE

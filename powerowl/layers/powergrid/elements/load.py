from typing import List

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .enums.connection_type import ConnectionType
from .enums.voltage_niveau import VoltageNiveau
from .grid_asset import GridAsset
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class Load(GridAsset):
    prefix = "load"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False, pp_column="name"),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("profile_name", Gvc.GENERIC, str, None, required=False),

            As("bus", Gvc.PROPERTY, GridNode, None),
            As("connection_type", Gvc.PROPERTY, ConnectionType, ConnectionType.NONE, required=False),
            As("constant_impedance", Gvc.PROPERTY, float, 0, Unit.PERCENT, pp_column="const_z_percent"),
            As("constant_current", Gvc.PROPERTY, float, 0, Unit.PERCENT, pp_column="const_i_percent"),
            As("nominal_power", Gvc.PROPERTY, float, None, Unit.VOLT_AMPERE, Scale.BASE, required=False),
            As("connection_type", Gvc.PROPERTY, ConnectionType, ConnectionType.NONE, required=False),
            As("controllable", Gvc.PROPERTY, bool, False, required=False),
            As("maximum_active_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("minimum_active_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("maximum_reactive_power", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),
            As("minimum_reactive_power", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),

            As("profile_enabled", Gvc.CONFIGURATION, bool, False, required=False, operator_controllable=False),
            As("active_power_profile_percentage", Gvc.CONFIGURATION, float, None, Unit.PERCENT, Scale.BASE, required=False, operator_controllable=False),
            As("reactive_power_profile_percentage", Gvc.CONFIGURATION, float, None, Unit.PERCENT, Scale.BASE, required=False, operator_controllable=False),

            As("connected", Gvc.CONFIGURATION, bool, True, related=[(Gvc.MEASUREMENT, "is_connected")], targets=[(Gvc.MEASUREMENT, "is_connected")]),
            As("scaling", Gvc.CONFIGURATION, float, 1),
            As("in_service", Gvc.CONFIGURATION, bool, True, operator_controllable=False),
            As("target_active_power", Gvc.CONFIGURATION, float, np.NAN, Unit.WATT, Scale.BASE, related=[(Gvc.MEASUREMENT, "active_power")]),
            As("target_reactive_power", Gvc.CONFIGURATION, float, np.NAN, Unit.VAR, Scale.BASE, related=[(Gvc.MEASUREMENT, "reactive_power")]),

            As("is_connected", Gvc.MEASUREMENT, bool, True, simulation_context=False, source=(Gvc.CONFIGURATION, "connected")),
            As("active_power", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE, related=[(Gvc.CONFIGURATION, "target_active_power")]),
            As("reactive_power", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE, related=[(Gvc.CONFIGURATION, "target_reactive_power")])
        ]


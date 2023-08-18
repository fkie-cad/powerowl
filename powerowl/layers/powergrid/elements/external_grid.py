from typing import List

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .grid_asset import GridAsset
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class ExternalGrid(GridAsset):
    prefix = "external_grid"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False, pp_column="name"),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("bus", Gvc.PROPERTY, GridNode, None),
            As("maximum_active_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("minimum_active_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("maximum_reactive_power", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),
            As("minimum_reactive_power", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),
            As("maximum_short_circuit_power_provision", Gvc.PROPERTY, float, None, Unit.VOLT_AMPERE, Scale.MEGA, required=False),
            As("minimum_short_circuit_power_provision", Gvc.PROPERTY, float, None, Unit.VOLT_AMPERE, Scale.MEGA, required=False),
            As("maximum_short_circuit_impedance_rx_ratio", Gvc.PROPERTY, float, None, required=False),
            As("minimum_short_circuit_impedance_rx_ratio", Gvc.PROPERTY, float, None, required=False),
            As("maximum_zero_sequence_rx_ratio", Gvc.PROPERTY, float, None, required=False),
            As("maximum_zero_sequence_x0x_ratio", Gvc.PROPERTY, float, None, required=False),

            As("in_service", Gvc.CONFIGURATION, bool, True),
            As("target_voltage", Gvc.CONFIGURATION, float, np.NAN, Unit.PER_UNIT),
            As("target_voltage_angle", Gvc.CONFIGURATION, float, np.NAN, Unit.DEGREE, Scale.BASE),

            As("active_power", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE),
            As("reactive_power", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE)
        ]

from typing import List

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .grid_asset import GridAsset
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class Motor(GridAsset):
    prefix = "motor"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("bus", Gvc.PROPERTY, GridNode, None),
            As("rated_voltage", Gvc.PROPERTY, float, None, Unit.VOLT, Scale.BASE, required=False),
            As("rated_mechanical_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE),
            As("rated_power_efficiency", Gvc.PROPERTY, float, np.NAN, Unit.PERCENT),
            As("rated_power_cos_phi", Gvc.PROPERTY, float, np.NAN),
            As("locked_rotor_current", Gvc.PROPERTY, float, np.NAN, Unit.PER_UNIT),
            As("controllable", Gvc.PROPERTY, bool, True),

            As("cos_phi", Gvc.CONFIGURATION, float, np.NAN),
            As("efficiency", Gvc.CONFIGURATION, float, 100, Unit.PERCENT),
            As("loading", Gvc.CONFIGURATION, float, 100, Unit.PERCENT),
            As("scaling", Gvc.CONFIGURATION, float, 1),
            As("in_service", Gvc.CONFIGURATION, bool, True, operator_controllable=False),

            As("active_power", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE),
            As("reactive_power", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE)
        ]

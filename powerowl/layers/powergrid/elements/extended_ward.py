from typing import List

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .grid_asset import GridAsset
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class ExtendedWard(GridAsset):
    prefix = "extended_ward"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("bus", Gvc.PROPERTY, GridNode, None),
            As("controllable", Gvc.PROPERTY, bool, True),
            As("resistance", Gvc.PROPERTY, float, np.NAN, Unit.OHM, Scale.BASE),
            As("reactance", Gvc.PROPERTY, float, np.NAN, Unit.OHM, Scale.BASE),
            As("controllable", Gvc.PROPERTY, bool, True),
            As("impedance_active_power_demand", Gvc.PROPERTY, float, np.NAN, Unit.WATT, Scale.BASE),
            As("impedance_reactive_power_demand", Gvc.PROPERTY, float, np.NAN, Unit.VAR, Scale.BASE),

            As("target_voltage", Gvc.CONFIGURATION, float, np.NAN, Unit.PER_UNIT),
            As("active_power_demand", Gvc.CONFIGURATION, float, np.NAN, Unit.WATT, Scale.BASE),
            As("reactive_power_demand", Gvc.CONFIGURATION, float, np.NAN, Unit.VAR, Scale.BASE),

            As("active_power", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE),
            As("reactive_power", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE),
            As("voltage", Gvc.MEASUREMENT, float, np.NAN, Unit.PER_UNIT)
        ]

from typing import List

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .grid_asset import GridAsset
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class Storage(GridAsset):
    prefix = "storage"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("bus", Gvc.PROPERTY, GridNode, None),
            As("type", Gvc.PROPERTY, str, "battery"),
            As("controllable", Gvc.PROPERTY, bool, True, required=False),
            As("nominal_power", Gvc.PROPERTY, float, None, Unit.VOLT_AMPERE, Scale.BASE, required=False),
            As("maximum_charge", Gvc.PROPERTY, float, None, Unit.WATT_HOUR, Scale.BASE, required=False,
               pp_column="max_e_mwh"),
            As("minimum_charge", Gvc.PROPERTY, float, None, Unit.WATT_HOUR, Scale.BASE, required=False,
               pp_column="min_e_mwh"),
            As("maximum_active_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("minimum_active_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("maximum_reactive_power", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),
            As("minimum_reactive_power", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),

            As("scaling", Gvc.CONFIGURATION, float, 1),
            As("in_service", Gvc.CONFIGURATION, bool, True),
            As("cos_phi", Gvc.CONFIGURATION, float, None, required=False),
            As("target_active_power", Gvc.CONFIGURATION, float, np.NAN, Unit.WATT, Scale.BASE),
            As("target_reactive_power", Gvc.CONFIGURATION, float, np.NAN, Unit.VAR, Scale.BASE),
            As("state_of_charge", Gvc.CONFIGURATION, float, 0, Unit.PERCENT, pp_column="soc_percent"),

            As("active_power", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE),
            As("reactive_power", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE),
            As("state_of_charge", Gvc.MEASUREMENT, float, 0, Unit.PERCENT)
        ]

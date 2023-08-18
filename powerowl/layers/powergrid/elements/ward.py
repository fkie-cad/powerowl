from typing import List

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .grid_asset import GridAsset
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class Ward(GridAsset):
    prefix = "ward"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("bus", Gvc.PROPERTY, GridNode, None),
            As("controllable", Gvc.PROPERTY, bool, False),
            As("impedance_active_power_demand", Gvc.PROPERTY, float, np.NAN, Unit.WATT, Scale.BASE, pp_column="pz_mw"),
            As("impedance_reactive_power_demand", Gvc.PROPERTY, float, np.NAN, Unit.VAR, Scale.BASE,
               pp_column="qz_mvar"),
            As("active_power_demand", Gvc.CONFIGURATION, float, np.NAN, Unit.WATT, Scale.BASE, pp_column="qs_mvar"),
            As("reactive_power_demand", Gvc.CONFIGURATION, float, np.NAN, Unit.VAR, Scale.BASE, pp_column="ps_mw"),

            As("in_service", Gvc.CONFIGURATION, bool, True, pp_column="in_service"),

            As("active_power", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE, pp_column="p_mw"),
            As("reactive_power", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE, pp_column="q_mvar"),
            As("voltage", Gvc.MEASUREMENT, float, np.NAN, Unit.PER_UNIT, pp_column="vm_pu")
        ]

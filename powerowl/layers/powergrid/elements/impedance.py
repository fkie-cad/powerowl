from typing import List

import numpy as np

from powerowl.layers.powergrid.elements.attribute_specification import AttributeSpecification as As
from powerowl.layers.powergrid.elements.grid_edge import GridEdge
from powerowl.layers.powergrid.elements.grid_node import GridNode
from powerowl.layers.powergrid.values.grid_value_context import GridValueContext as Gvc
from powerowl.layers.powergrid.values.units.scale import Scale
from powerowl.layers.powergrid.values.units.unit import Unit


class Impedance(GridEdge):
    prefix = "impedance"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("from_bus", Gvc.PROPERTY, GridNode, None),
            As("to_bus", Gvc.PROPERTY, GridNode, None),
            As("resistance_from_to", Gvc.PROPERTY, float, np.NAN, Unit.OHM, Scale.BASE),
            As("reactance_from_to", Gvc.PROPERTY, float, np.NAN, Unit.OHM, Scale.BASE),
            As("resistance_to_from", Gvc.PROPERTY, float, np.NAN, Unit.OHM, Scale.BASE),
            As("reactance_to_from", Gvc.PROPERTY, float, np.NAN, Unit.OHM, Scale.BASE),
            As("apparent_power", Gvc.PROPERTY, float, np.NAN, Unit.VOLT_AMPERE, Scale.BASE),

            As("in_service", Gvc.CONFIGURATION, bool, True),

            As("active_power_from", Gvc.MEASUREMENT, float, None, Unit.WATT, Scale.BASE, required=False),
            As("reactive_power_from", Gvc.MEASUREMENT, float, None, Unit.VAR, Scale.BASE, required=False),
            As("active_power_to", Gvc.MEASUREMENT, float, None, Unit.WATT, Scale.BASE, required=False),
            As("reactive_power_to", Gvc.MEASUREMENT, float, None, Unit.VAR, Scale.BASE, required=False),
            As("active_power_loss", Gvc.MEASUREMENT, float, None, Unit.WATT, Scale.BASE, required=False),
            As("reactive_power_loss", Gvc.MEASUREMENT, float, None, Unit.VAR, Scale.BASE, required=False),
            As("current_from", Gvc.MEASUREMENT, float, None, Unit.AMPERE, Scale.BASE, required=False),
            As("current_to", Gvc.MEASUREMENT, float, None, Unit.AMPERE, Scale.BASE, required=False)
        ]

from typing import List, Tuple

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .enums.line_type import LineType
from .grid_edge import GridEdge
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class Line(GridEdge):
    prefix = "line"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False, pp_column="name"),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("from_bus", Gvc.PROPERTY, GridNode, None),
            As("to_bus", Gvc.PROPERTY, GridNode, None),
            As("type", Gvc.PROPERTY, LineType, LineType.NONE),
            As("type_name", Gvc.PROPERTY, str, None, required=False, pp_column="std_type"),
            As("length", Gvc.PROPERTY, float, None, Unit.METER, Scale.BASE, pp_column="length_km"),
            As("maximum_current", Gvc.PROPERTY, float, None, Unit.AMPERE, Scale.BASE, pp_column="max_i_ka"),
            As("resistance_per_km", Gvc.PROPERTY, float, None, Unit.OHM, Scale.BASE, pp_column="r_ohm_per_km"),
            As("reactance_per_km", Gvc.PROPERTY, float, None, Unit.OHM, Scale.BASE, pp_column="x_ohm_per_km"),
            As("capacitance_per_km", Gvc.PROPERTY, float, None, Unit.FARAD, Scale.BASE, pp_column="c_nf_per_km"),
            As("zero_sequence_resistance_per_km", Gvc.PROPERTY, float, np.NAN, Unit.OHM, Scale.BASE,
               pp_column="r0_ohm_per_km", required=False),
            As("zero_sequence_reactance_per_km", Gvc.PROPERTY, float, np.NAN, Unit.OHM, Scale.BASE,
               pp_column="x0_ohm_per_km", required=False),
            As("zero_sequence_capacitance_per_km", Gvc.PROPERTY, float, np.NAN, Unit.FARAD, Scale.BASE,
               pp_column="c0_nf_per_km", required=False),
            As("position-sequence", Gvc.PROPERTY, List[Tuple[float, float]], None, required=False,
               pp_column="geodata"),

            As("active_power_from", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE, pp_column="p_from_mw"),
            As("reactive_power_from", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE, pp_column="q_from_mvar"),
            As("active_power_to", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE, pp_column="p_to_mw"),
            As("reactive_power_to", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE, pp_column="q_to_mvar"),
            As("active_power_loss", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE, pp_column="pl_mw"),
            As("reactive_power_loss", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE, pp_column="ql_mvar"),
            As("current_from", Gvc.MEASUREMENT, float, np.NAN, Unit.AMPERE, Scale.BASE, pp_column="i_from_ka"),
            As("current_to", Gvc.MEASUREMENT, float, np.NAN, Unit.AMPERE, Scale.BASE, pp_column="i_to_ka"),
            As("current", Gvc.MEASUREMENT, float, np.NAN, Unit.AMPERE, Scale.BASE, pp_column="i_ka"),
            As("voltage_from", Gvc.MEASUREMENT, float, np.NAN, Unit.PER_UNIT, pp_column="vm_from_pu"),
            As("voltage_to", Gvc.MEASUREMENT, float, np.NAN, Unit.PER_UNIT, pp_column="vm_to_pu"),
            As("voltage_angle_from", Gvc.MEASUREMENT, float, np.NAN, Unit.DEGREE, pp_column="va_from_degree"),
            As("voltage_angle_to", Gvc.MEASUREMENT, float, np.NAN, Unit.DEGREE, pp_column="va_to_degree"),
            As("loading", Gvc.MEASUREMENT, float, np.NAN, Unit.PERCENT, pp_column="loading_percent"),
        ]

    def get_bus_a(self):
        return self.get_from_bus()

    def get_bus_b(self):
        return self.get_to_bus()

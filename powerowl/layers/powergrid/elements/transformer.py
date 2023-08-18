from typing import List

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .enums.tap_side import TapSide
from .enums.vector_group import VectorGroup
from .grid_edge import GridEdge
from .grid_node import GridNode
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class Transformer(GridEdge):
    prefix = "trafo"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False, pp_column="name"),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("hv_bus", Gvc.PROPERTY, GridNode, None),
            As("lv_bus", Gvc.PROPERTY, GridNode, None),
            As("controllable", Gvc.PROPERTY, bool, True, required=False),
            As("std_type", Gvc.PROPERTY, str, None, required=False, pp_column="std_type"),
            As("nominal_power", Gvc.PROPERTY, float, np.NAN, Unit.VOLT_AMPERE, Scale.BASE),
            As("voltage_niveau_hv", Gvc.PROPERTY, float, None, Unit.VOLT, Scale.BASE),
            As("voltage_niveau_lv", Gvc.PROPERTY, float, None, Unit.VOLT, Scale.BASE),
            As("short_circuit_voltage", Gvc.PROPERTY, float, None, Unit.PERCENT, pp_column="vk_percent"),
            As("short_circuit_voltage_real", Gvc.PROPERTY, float, None, Unit.PERCENT, pp_column="vkr_percent"),
            As("iron_power_loss", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, pp_column="pfe_kw"),
            As("open_loop_loss", Gvc.PROPERTY, float, None, Unit.PERCENT, pp_column="i0_percent"),
            As("zero_sequence_short_circuit_voltage", Gvc.PROPERTY, float, np.NAN, Unit.PERCENT,
               required=False, pp_column="vk0_percent"),
            As("zero_sequence_short_circuit_voltage_real", Gvc.PROPERTY, float, np.NAN, Unit.PERCENT,
               required=False, pp_column="vkr0_percent"),
            As("magnetizing_ration", Gvc.PROPERTY, float, np.NAN, Unit.PERCENT,
               pp_column="mag0_percent", required=False),
            As("zero_sequence_magnetizing_ration", Gvc.PROPERTY, float, np.NAN, Unit.PERCENT,
               required=False, pp_column="mag0_rx"),
            As("zero_sequence_short_circuit_impedance", Gvc.PROPERTY, float, np.NAN,
               required=False, pp_column="si0_hv_partial"),
            As("vector_group", Gvc.PROPERTY, VectorGroup, VectorGroup.NONE,
               required=False, pp_column="vector_group"),
            As("phase_shift", Gvc.PROPERTY, float, 0, Unit.DEGREE, pp_column="shift_degree"),
            As("tap_side", Gvc.PROPERTY, TapSide, TapSide.NONE, pp_column="tap_side"),
            As("tap_neutral", Gvc.PROPERTY, int, np.NAN, pp_column="tap_neutral"),
            As("tap_minimum", Gvc.PROPERTY, int, np.NAN, pp_column="tap_min"),
            As("tap_maximum", Gvc.PROPERTY, int, np.NAN, pp_column="tap_max"),
            As("tap_phase_shifter", Gvc.PROPERTY, bool, False, pp_column="tap_phase_shifter"),
            As("voltage_tap_step", Gvc.PROPERTY, int, np.NAN, unit=Unit.PERCENT, pp_column="tap_step_percent"),
            As("angle_tap_step", Gvc.PROPERTY, int, np.NAN, unit=Unit.DEGREE, pp_column="tap_step_degree"),
            As("parallel", Gvc.PROPERTY, int, 1, pp_column="parallel"),
            As("maximum_load", Gvc.PROPERTY, float, np.NAN, unit=Unit.PERCENT,
               required=False, pp_column="max_loading_percent"),

            As("tap_position", Gvc.CONFIGURATION, int, np.NAN, pp_column="tap_pos"),
            As("in_service", Gvc.CONFIGURATION, bool, True),

            As("active_power_hv", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE),
            As("reactive_power_hv", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE),
            As("active_power_lv", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE),
            As("reactive_power_lv", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE),
            As("active_power_loss", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE),
            As("reactive_power_loss", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE),
            As("current_hv", Gvc.MEASUREMENT, float, np.NAN, Unit.AMPERE, Scale.BASE),
            As("current_lv", Gvc.MEASUREMENT, float, np.NAN, Unit.AMPERE, Scale.BASE),
            As("voltage_hv", Gvc.MEASUREMENT, float, np.NAN, Unit.PER_UNIT),
            As("voltage_lv", Gvc.MEASUREMENT, float, np.NAN, Unit.PER_UNIT),
            As("voltage_angle_hv", Gvc.MEASUREMENT, float, np.NAN, Unit.DEGREE),
            As("voltage_angle_lv", Gvc.MEASUREMENT, float, np.NAN, Unit.DEGREE),
            As("loading", Gvc.MEASUREMENT, float, np.NAN, Unit.PERCENT)
        ]

    def get_hv_bus(self):
        return self.get_property_value("hv_bus")

    def get_lv_bus(self):
        return self.get_property_value("lv_bus")

    def get_bus_a(self):
        return self.get_hv_bus()

    def get_bus_b(self):
        return self.get_lv_bus()

    def get_from_bus(self):
        return self.get_hv_bus()

    def get_to_bus(self):
        return self.get_lv_bus()

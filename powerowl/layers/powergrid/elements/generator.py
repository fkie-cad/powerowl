from typing import List

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .enums.generator_type import GeneratorType
from .enums.reactive_power_mode import ReactivePowerMode
from .grid_asset import GridAsset
from .grid_node import GridNode
from .transformer import Transformer
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class Generator(GridAsset):
    prefix = "gen"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_specifications() -> List[As]:
        return [
            As("name", Gvc.GENERIC, str, None, required=False),
            As("observable", Gvc.GENERIC, bool, True, required=False),

            As("bus", Gvc.PROPERTY, GridNode, None),
            As("type", Gvc.PROPERTY, GeneratorType, GeneratorType.NONE),
            As("controllable", Gvc.PROPERTY, bool, True, required=False),
            As("nominal_power", Gvc.PROPERTY, float, None, Unit.VOLT_AMPERE, Scale.BASE, required=False),
            As("minimum_active_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("minimum_reactive_power", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),
            As("maximum_active_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("maximum_reactive_power", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),
            As("rated_voltage", Gvc.PROPERTY, float, None, Unit.VOLT, Scale.BASE, required=False),
            As("transformer", Gvc.PROPERTY, Transformer, None, required=False),

            # Limits can be used to apply, e.g., solar panel or wind turbine, limits.
            As("active_power_limit", Gvc.CONFIGURATION, float, None, Unit.WATT, Scale.BASE, required=False, operator_controllable=False),
            As("reactive_power_limit", Gvc.CONFIGURATION, float, None, Unit.VAR, Scale.BASE, required=False, operator_controllable=False),

            As("profile_enabled", Gvc.CONFIGURATION, bool, False, required=False, operator_controllable=False),
            As("explicit_control", Gvc.CONFIGURATION, bool, False, required=False, operator_controllable=True,
               targets=[(Gvc.CONFIGURATION, "profile_enabled", lambda b: not b)]),
            As("active_power_profile_percentage", Gvc.CONFIGURATION, float, None, Unit.PERCENT, Scale.BASE, required=False, operator_controllable=False),

            As("scaling", Gvc.CONFIGURATION, float, 1),
            As("connected", Gvc.CONFIGURATION, bool, True, related=[(Gvc.MEASUREMENT, "is_connected")]),
            As("in_service", Gvc.CONFIGURATION, bool, True, operator_controllable=False),

            As("target_active_power_percentage", Gvc.CONFIGURATION, float, 100, Unit.PERCENT, Scale.BASE, lower_bound=0, upper_bound=100),
            As("target_voltage", Gvc.CONFIGURATION, float, None, Unit.PER_UNIT, required=False, related=[(Gvc.MEASUREMENT, "voltage")]),
            As("target_active_power", Gvc.CONFIGURATION, float, np.NAN, Unit.WATT, Scale.BASE, related=[(Gvc.MEASUREMENT, "active_power")]),

            As("is_connected", Gvc.MEASUREMENT, bool, True, simulation_context=False, related=[(Gvc.CONFIGURATION, "connected")]),
            As("cos_phi", Gvc.MEASUREMENT, float, np.NAN, Unit.NONE, Scale.BASE, simulation_context=False),
            As("active_power", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE, related=[(Gvc.CONFIGURATION, "target_active_power")]),
            As("reactive_power", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE),
            As("voltage", Gvc.MEASUREMENT, float, np.NAN, Unit.PER_UNIT, related=[(Gvc.CONFIGURATION, "target_voltage")]),
            As("voltage_angle", Gvc.MEASUREMENT, float, np.NAN, Unit.DEGREE),

            As("profile_percentage", Gvc.MEASUREMENT, float, 100, Unit.PERCENT, Scale.BASE, source=(Gvc.CONFIGURATION, "active_power_profile_percentage"))
        ]

    def get_allowed_properties(self) -> List[str]:
        return ["name", "bus", "type", "sn_mva", "type", "controllable",
                "max_p_mw", "max_q_mvar", "min_p_mw", "min_q_mvar",
                "vn_kn", "xdss_pu", "rdss_ohm"]

    def get_allowed_config(self) -> List[str]:
        return ["in_service", "p_mw", "q_mvar", "scaling", "vm_pu", "cos_phi"]

    def get_allowed_measurements(self) -> List[str]:
        return ["p_mw", "q_mvar", "va_degree", "vm_pu"]

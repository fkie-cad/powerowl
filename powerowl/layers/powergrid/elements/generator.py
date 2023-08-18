from typing import List

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .enums.generator_type import GeneratorType
from .grid_element import GridElement
from .grid_node import GridNode
from .transformer import Transformer
from ..values.grid_value_context import GridValueContext as Gvc
from ..values.units.scale import Scale
from ..values.units.unit import Unit


class Generator(GridElement):
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
            As("controllable", Gvc.PROPERTY, bool, True),
            As("nominal_power", Gvc.PROPERTY, float, None, Unit.VOLT_AMPERE, Scale.BASE, required=False),
            As("maximum_active_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("minimum_active_power", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),
            As("maximum_reactive_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("minimum_reactive_power", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),
            As("rated_voltage", Gvc.PROPERTY, float, None, Unit.VOLT, Scale.BASE, required=False),
            As("transformer", Gvc.PROPERTY, Transformer, None, required=False),

            As("scaling", Gvc.CONFIGURATION, float, 1),
            As("in_service", Gvc.CONFIGURATION, bool, True),
            As("cos_phi", Gvc.CONFIGURATION, float, None, required=False),
            As("target_voltage", Gvc.CONFIGURATION, float, None, Unit.PER_UNIT, required=False),
            As("target_active_power", Gvc.CONFIGURATION, float, np.NAN, Unit.WATT, Scale.BASE),
            As("target_reactive_power", Gvc.CONFIGURATION, float, np.NAN, Unit.VAR, Scale.BASE),

            As("active_power", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE),
            As("reactive_power", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE),
            As("voltage", Gvc.MEASUREMENT, float, np.NAN, Unit.PER_UNIT),
            As("voltage_angle", Gvc.MEASUREMENT, float, np.NAN, Unit.DEGREE)
        ]

    def get_allowed_properties(self) -> List[str]:
        return ["name", "bus", "type", "sn_mva", "type", "controllable",
                "max_p_mw", "max_q_mvar", "min_p_mw", "min_q_mvar",
                "vn_kn", "xdss_pu", "rdss_ohm"]

    def get_allowed_config(self) -> List[str]:
        return ["in_service", "p_mw", "q_mvar", "scaling", "vm_pu", "cos_phi"]

    def get_allowed_measurements(self) -> List[str]:
        return ["p_mw", "q_mvar", "va_degree", "vm_pu"]

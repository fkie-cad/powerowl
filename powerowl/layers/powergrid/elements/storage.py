from typing import List

import numpy as np

from .attribute_specification import AttributeSpecification as As
from .enums.reactive_power_mode import ReactivePowerMode
from .enums.storage_type import StorageType
from .enums.voltage_niveau import VoltageNiveau
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

            As("profile_name", Gvc.GENERIC, str, None, required=False),

            As("bus", Gvc.PROPERTY, GridNode, None),
            As("storage_type", Gvc.PROPERTY, StorageType, StorageType.NONE),
            As("controllable", Gvc.PROPERTY, bool, True, required=False),
            As("nominal_power", Gvc.PROPERTY, float, None, Unit.VOLT_AMPERE, Scale.BASE, required=False),
            As("maximum_charge", Gvc.PROPERTY, float, None, Unit.WATT_HOUR, Scale.BASE, required=False, pp_column="max_e_mwh"),
            As("minimum_charge", Gvc.PROPERTY, float, None, Unit.WATT_HOUR, Scale.BASE, required=False, pp_column="min_e_mwh"),
            As("maximum_active_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("minimum_active_power", Gvc.PROPERTY, float, None, Unit.WATT, Scale.BASE, required=False),
            As("maximum_reactive_power", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),
            As("minimum_reactive_power", Gvc.PROPERTY, float, None, Unit.VAR, Scale.BASE, required=False),

            As("profile_enabled", Gvc.CONFIGURATION, bool, False, required=False, operator_controllable=False),
            As("explicit_control", Gvc.CONFIGURATION, bool, False, required=False, operator_controllable=True,
               targets=[(Gvc.CONFIGURATION, "profile_enabled", lambda b: not b)]),

            As("active_power_profile_percentage", Gvc.CONFIGURATION, float, None, Unit.PERCENT, Scale.BASE, required=False, operator_controllable=False),
            As("reactive_power_profile_percentage", Gvc.CONFIGURATION, float, None, Unit.PERCENT, Scale.BASE, required=False, operator_controllable=False),

            As("connected", Gvc.CONFIGURATION, bool, True, targets=[(Gvc.MEASUREMENT, "is_connected")]),
            As("scaling", Gvc.CONFIGURATION, float, 1, operator_controllable=False),
            As("in_service", Gvc.CONFIGURATION, bool, True, operator_controllable=False),
            As("reactive_power_mode", Gvc.CONFIGURATION, ReactivePowerMode, ReactivePowerMode.MANUAL, simulation_context=False, operator_controllable=False),

            As("target_cos_phi", Gvc.CONFIGURATION, float, None, simulation_context=False, lower_bound=0, upper_bound=1),
            As("target_active_power_percentage", Gvc.CONFIGURATION, float, 100, Unit.PERCENT, Scale.BASE, lower_bound=-100, upper_bound=100),
            As("target_active_power", Gvc.CONFIGURATION, float, np.NAN, Unit.WATT, Scale.BASE, related=[(Gvc.MEASUREMENT, "active_power")]),
            As("target_reactive_power", Gvc.CONFIGURATION, float, np.NAN, Unit.VAR, Scale.BASE, related=[(Gvc.MEASUREMENT, "reactive_power")]),
            As("state_of_charge", Gvc.CONFIGURATION, float, 0, Unit.PERCENT, related=[(Gvc.MEASUREMENT, "state_of_charge")], operator_controllable=False),
            As("current_charge", Gvc.CONFIGURATION, float, 0, Unit.WATT_HOUR, related=[(Gvc.MEASUREMENT, "current_charge")], operator_controllable=False),

            As("is_connected", Gvc.MEASUREMENT, bool, True, simulation_context=False, source=(Gvc.CONFIGURATION, "connected")),
            As("cos_phi", Gvc.MEASUREMENT, float, np.NAN, Unit.NONE, Scale.BASE, simulation_context=False, related=[(Gvc.CONFIGURATION, "target_cos_phi")]),
            As("active_power", Gvc.MEASUREMENT, float, np.NAN, Unit.WATT, Scale.BASE, related=[(Gvc.CONFIGURATION, "target_active_power")]),
            As("reactive_power", Gvc.MEASUREMENT, float, np.NAN, Unit.VAR, Scale.BASE, related=[(Gvc.CONFIGURATION, "target_reactive_power")]),
            As("state_of_charge", Gvc.MEASUREMENT, float, 0, Unit.PERCENT, source=(Gvc.CONFIGURATION, "state_of_charge")),
            As("current_charge", Gvc.MEASUREMENT, float, 0, Unit.WATT_HOUR, source=(Gvc.CONFIGURATION, "current_charge"))
        ]

    def get_voltage_niveau(self) -> VoltageNiveau:
        s_type: StorageType = self.get_property_value("storage_type")
        return s_type.get_voltage_niveau()
